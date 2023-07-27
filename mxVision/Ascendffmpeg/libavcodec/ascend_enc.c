/*
 * Copyright(c) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except int compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "ascend_enc.h"

static void CloseEpoll(int32_t epollFd)
{
    int ret = hi_mpi_sys_close_epoll(epollFd);
    if (ret != 0) {
        av_log(NULL, AV_LOG_ERROR, "Call hi_mpi_sys_close_epoll failed, ret is %d.\n", ret);
    }
}

static int get_stream_loop(ASCENDEncContext_t *ctx, int32_t epollFd)
{
    int i = 0;
    int eos_flag = 0;
    while (ctx->thread_run_flag) {
        if (ctx->eos_post_flag == 1 && (ctx->frame_send_sum <= i)) {
            // eos
            StreamInfo_t stream_info;
            memset(&stream_info, 0, sizeof(StreamInfo_t));
            stream_info.event_type = EVENT_EOS;
            
            ff_mutex_lock(&ctx->queue_mutex);
            av_fifo_generic_write(ctx->frame_queue, &stream_info, sizeof(StreamInfo_t), NULL);
            ff_mutex_unlock(&ctx->queue_mutex);
            sem_post(&ctx->eos_sema);
            av_log(ctx, AV_LOG_DEBUG, "Encoder got eos.\n");
            break;
        }

        int32_t eventCount = 0;
        hi_dvpp_epoll_event events[HI_DVPP_EPOLL_EVENT];
        int ret = hi_mpi_sys_wait_epoll(epollFd, events, HI_MPI_SYS_WAIT_EPOLL_MAX_EVENTS,
                                        HI_DVPP_EPOLL_EVENT_NUM, &eventCount);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_wait_epoll failed, ret is %d.\n", ret);
            return -1;
        }

        hi_venc_chn_status stat;
        ret = hi_mpi_venc_query_status(ctx->channel_id, &stat);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_query_status failed, ret is %d.\n", ret);
            return -1;
        }

        hi_venc_stream stream;
        stream.pack_cnt = stat.cur_packs;

        hi_venc_pack pack[MAX_PACK_COUNT];
        stream.pack = pack;
        if (stream.pack == NULL) {
            return -1;
        }

        ret = hi_mpi_venc_get_stream(ctx->channel_id, &stream, WAIT_GET_TILL_TIMEOUT);
        if (ret != 0) {
            if (ctx->encoder_flushing && ret == HI_ERR_VENC_BUF_EMPTY) {
                eos_flag = 1;
                av_log(ctx, AV_LOG_DEBUG, "Encoder_flushing or stream eos.\n");
            } else {
                av_log(ctx, AV_LOG_DEBUG, "Call hi_mpi_venc_get_stream failed, ret is %d.\n", ret);
                continue;
            }
        }

        if (eos_flag) {
            //eos
            StreamInfo_t stream_info;
            memset(&stream_info, 0, sizeof(StreamInfo_t));
            stream_info.event_type = EVENT_EOS;

            ff_mutex_lock(&ctx->queue_mutex);
            av_fifo_generic_write(ctx->frame_queue, &stream_info, sizeof(StreamInfo_t), NULL);
            ff_mutex_unlock(&ctx->queue_mutex);
            sem_post(&ctx->eos_sema);
            av_log(ctx, AV_LOG_DEBUG, "Encoder got eos.\n");
            break;
        }

        // Create stream.
        uint32_t streamSize = stream.pack[0].len - stream.pack[0].offset;

        // Make sure stream size is bigger than 0'
        if (stream.pack[0].len <= stream.pack[0].offset) {
            av_log(ctx, AV_LOG_ERROR, "StreamSize(%d) is invalid, it has to be bigger than 0.\n",
                   stream.pack[0].len - stream.pack[0].offset);
            return -1;
        }

        // Make sure sream size is less than 2 Gigabytes
        if (streamSize > MAX_MEMORY_SIZE) {
            av_log(ctx, AV_LOG_ERROR, "StreamSize(%d) is invalid, it has to be less than 2G.\n", streamSize);
            return -1;
        }

        StreamInfo_t stream_info;
        stream_info.ascend_ctx = ctx;
        get_venc_stream_info(&stream_info, stream);
        uint8_t* data_frame = NULL;

        ff_mutex_lock(&ctx->queue_mutex);
        av_fifo_generic_read(ctx->dataptr_queue, &data_frame, sizeof(uint8_t*), NULL);
        av_fifo_generic_write(ctx->frame_queue, &stream_info, sizeof(StreamInfo_t), NULL);
        ff_mutex_unlock(&ctx->queue_mutex);

        ret = hi_mpi_venc_release_stream(ctx->channel_id, &stream);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_release_stream failed, ret is %d.\n", ret);
            return -1;
        }

        ret = hi_mpi_dvpp_free(data_frame);
        if (ret != 0) {
            av_log(ctx, AV_LOG_WARNING, "Call hi_mpi_dvpp_free failed, ret is %d.\n", ret);
        }

        i++;
        av_log(ctx, AV_LOG_DEBUG, "Finish getting the %d th frame.\n", i);
    }
    av_log(ctx, AV_LOG_DEBUG, "Encode thread get eos signal.\n");
    return 0;
}

static void *venc_get_stream(void *arg)
{
    ASCENDEncContext_t *ctx = (ASCENDEncContext_t*)arg;

    // Set Device
    av_log(NULL, AV_LOG_INFO, "Encode thread start.\n");
    int eos_flag = 0;
    int ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed at line(%d) in func(%s), ret is %d.\n", __LINE__, __func__, ret);
        return ((void*) (-1));
    }

    // init 
    int32_t epollFd = 0;
    int32_t fd = hi_mpi_venc_get_fd(ctx->channel_id);
    ret = hi_mpi_sys_create_epoll(HI_SYS_CREATE_EPOLL_SIZE, &epollFd);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_create_epoll failed, ret is %d.\n", ret);
        return ((void*) (-1));
    }
    hi_dvpp_epoll_event event;
    event.events = HI_DVPP_EPOLL_IN;
    event.data = (void*)(uint64_t)(fd);
    ret = hi_mpi_sys_ctl_epoll(epollFd, HI_DVPP_EPOLL_CTL_ADD, fd, &event);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_ctl_epoll add failed, ret is %d.\n", ret);
        CloseEpoll(epollFd);
        return ((void*) (-1));
    }

    // start reveive loop
    ret = get_stream_loop(ctx, epollFd);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call get_stream_loop failed, ret is %d.\n", ret);
        CloseEpoll(epollFd);
        return ((void*) (-1));
    }

    ret = hi_mpi_sys_ctl_epoll(epollFd, HI_DVPP_EPOLL_CTL_DEL, fd, NULL);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_ctl_epoll del failed, ret is %d.\n", ret);
        CloseEpoll(epollFd);
        return ((void*) (-1));
    }

    CloseEpoll(epollFd);

    return NULL;
}

static inline int encode_params_checking(AVCodecContext* avctx)
{
    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            if (avctx->width * avctx->height > 4096 * 2304) {
                av_log(avctx, AV_LOG_ERROR,
                       "ASCEND encoder H264 input max pixel num should be less than 4096x2304"
                       ", now: %dx%d.\n", avctx->width, avctx->height);
                return -1;
            }
        case AV_CODEC_ID_H265:
            if (avctx->width < 128 || avctx->width > 4096 || avctx->height < 128 || avctx->height > 4096) {
                av_log(avctx, AV_LOG_ERROR,
                       "ASCEND encoder only support size: 128x128~4096x4096, now: %dx%d.\n",
                       avctx->width, avctx->height);
                return -1;
            }
            break;
        default:
            break;
    }
    return 0;
}

static inline int set_venc_mod_param(AVCodecContext *avctx)
{
    hi_venc_mod_param mod_param;

    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            mod_param.mod_type = HI_VENC_MOD_H264;
            break;
        case AV_CODEC_ID_H265:
            mod_param.mod_type = HI_VENC_MOD_H265;
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Ascend encoder only support h264 or h265.\n");
            return AVERROR_BUG;
            break;
    }

    int ret = hi_mpi_venc_get_mod_param(&mod_param);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "Call hi_mpi_venc_get_mod_param failed, ret is: %d.\n", ret);
        return ret;
    }

    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
            mod_param.h264_mod_param.one_stream_buf = 1;
            break;
        case AV_CODEC_ID_H265:
            mod_param.h265_mod_param.one_stream_buf = 1;
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Ascend encoder only support h264 or h265.\n");
            return AVERROR_BUG;
            break;
    }

    ret = hi_mpi_venc_set_mod_param(&mod_param);
    if (ret != 0 && (unsigned int)ret != HI_ERR_VENC_NOT_PERM) {
        av_log(avctx, AV_LOG_ERROR, "Call hi_mpi_venc_set_mod_param failed, ret is %d.\n", ret);
        return ret;
    } else if ((unsigned int)ret == HI_ERR_VENC_NOT_PERM) {
        av_log(avctx, AV_LOG_WARNING, "Venc channel already exists, using default mod param.\n");
    }

    return 0;
}

static inline int create_venc_channel_by_order(ASCENDEncContext_t *ctx)
{
    int chnId = 0;
    int ret;
    while(1) {
        ret = hi_mpi_venc_create_chn(chnId, &ctx->chn_attr_);
        if (ret == 0) {
            ctx->channel_id = chnId;
            av_log(ctx, AV_LOG_WARNING, "The specified channel is occupied now, another channle will be arranged.\n");
            av_log(ctx, AV_LOG_INFO, "Create venc channels success, channel id is %d.\n", ctx->channel_id);
            break;
        } else if ((unsigned int)ret == HI_ERR_VENC_EXIST) {
            chnId++;
            if (chnId > MAX_HIMPI_VENC_CHN_NUM) {
                av_log(ctx, AV_LOG_ERROR, "All venc channels were occupied, create venc channel failed.\n");
                return -1;
            }
        } else {
            av_log(ctx, AV_LOG_ERROR, "Failed to create venc channel, ret is %d.\n", ret);
            return ret;
        }
    }
    return 0;
}

static inline int create_venc_channel(ASCENDEncContext_t *ctx)
{
    ctx->chn_attr_.venc_attr.max_pic_width = MAX_VENC_WIDTH;
    ctx->chn_attr_.venc_attr.max_pic_height = MAX_VENC_HEIGHT;
    ctx->chn_attr_.venc_attr.pic_width = ctx->coded_width;
    ctx->chn_attr_.venc_attr.pic_height = ctx->coded_height;
    if (ctx->is_movement_scene == 1) {
        ctx->chn_attr_.venc_attr.buf_size = FFALIGN(MAX_VENC_WIDTH * MAX_VENC_HEIGHT * BUF_SIZE_TIMES, BUF_SIZE_STRIDE);
    } else {
        ctx->chn_attr_.venc_attr.buf_size = MAX_VENC_WIDTH * MAX_VENC_HEIGHT * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    }

    ctx->chn_attr_.venc_attr.is_by_frame = HI_TRUE;
    int stats_time = (ctx->is_movement_scene != 0) ? (int)(1.0 * ctx->gop / ctx->frame_rate + 0.5) : HI_VENC_CHN_ATTR_STATS_TIME;

    if (ctx->avctx->codec->id == AV_CODEC_ID_H264) {
        ctx->chn_attr_.venc_attr.type = HI_PT_H264;
        ctx->chn_attr_.venc_attr.profile = ctx->profile;

        if (ctx->rc_mode == 1) {
            // VBR
            ctx->chn_attr_.rc_attr.rc_mode = HI_VENC_RC_MODE_H264_VBR;
            ctx->chn_attr_.rc_attr.h264_vbr.gop = ctx->gop;
            ctx->chn_attr_.rc_attr.h264_vbr.stats_time = stats_time;
            ctx->chn_attr_.rc_attr.h264_vbr.src_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h264_vbr.dst_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h264_vbr.max_bit_rate = ctx->max_bit_rate;
        } else {
            // CBR
            ctx->chn_attr_.rc_attr.rc_mode = HI_VENC_RC_MODE_H264_CBR;
            ctx->chn_attr_.rc_attr.h264_cbr.gop = ctx->gop;
            ctx->chn_attr_.rc_attr.h264_cbr.stats_time = stats_time;
            ctx->chn_attr_.rc_attr.h264_cbr.src_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h264_cbr.dst_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h264_cbr.bit_rate = ctx->max_bit_rate;
        }
    } else if (ctx->avctx->codec->id == AV_CODEC_ID_H265) {
        ctx->chn_attr_.venc_attr.type = HI_PT_H265;
        ctx->chn_attr_.venc_attr.profile = HI_VENC_H265_MAIN_LEVEL;

        if (ctx->rc_mode == 1) {
            // VBR
            ctx->chn_attr_.rc_attr.rc_mode = HI_VENC_RC_MODE_H265_VBR;
            ctx->chn_attr_.rc_attr.h265_vbr.gop = ctx->gop;
            ctx->chn_attr_.rc_attr.h265_vbr.stats_time = stats_time;
            ctx->chn_attr_.rc_attr.h265_vbr.src_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h265_vbr.dst_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h265_vbr.max_bit_rate = ctx->max_bit_rate;
        } else {
            // CBR
            ctx->chn_attr_.rc_attr.rc_mode = HI_VENC_RC_MODE_H265_CBR;
            ctx->chn_attr_.rc_attr.h265_cbr.gop = ctx->gop;
            ctx->chn_attr_.rc_attr.h265_cbr.stats_time = stats_time;
            ctx->chn_attr_.rc_attr.h265_cbr.src_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h265_cbr.dst_frame_rate = ctx->frame_rate;
            ctx->chn_attr_.rc_attr.h265_cbr.bit_rate = ctx->max_bit_rate;
        }
    } else {
        av_log(ctx, AV_LOG_ERROR, "Ascend encoder only support h264 or h265.\n");
        return AVERROR_BUG;
    }

    ctx->chn_attr_.gop_attr.gop_mode = HI_VENC_GOP_MODE_NORMAL_P;
    ctx->chn_attr_.gop_attr.normal_p.ip_qp_delta = HI_ODD_NUM_3;

    int ret = hi_mpi_venc_create_chn(ctx->channel_id, &ctx->chn_attr_);
    if (ret == 0) {
        av_log(ctx, AV_LOG_INFO, "Create venc channels success. Channel id is %d.\n", ctx->channel_id);
    } else if ((unsigned int)ret == HI_ERR_VENC_EXIST) {
        ret = create_venc_channel_by_order(ctx);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to create venc channel, ret is %d.\n", ret);
            return ret;
        }
    } else {
        av_log(ctx, AV_LOG_ERROR, "Failed to create venc channel, ret is %d.\n", ret);
        return ret;
    }

    hi_venc_scene_mode scene_mode = HI_VENC_SCENE_0;
    if (ctx->avctx->codec_id == AV_CODEC_ID_H265 && ctx->is_movement_scene != 0) {
        scene_mode = HI_VENC_SCENE_1;
    }
    ret = hi_mpi_venc_set_scene_mode(ctx->channel_id, scene_mode);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_set_scene_mode failed, ret is %d.\n", ret);
        return ret;
    }

    return 0;
}

static inline int set_venc_rc_param(ASCENDEncContext_t *ctx)
{
    int ret;
    hi_s32 i = 0;
    hi_u32 u32Thrd_ori[16] = {0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255};
    hi_venc_rc_param rcParam;
    ret = hi_mpi_venc_get_rc_param(ctx->channel_id, &rcParam);
    if (ret != HI_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_get_rc_param [%u] failed with ret 0x%x.\n", ctx->channel_id, ret);
        return ret;
    }
    for (i = 0; i < 16; i++) {
        hi_u32 value = u32Thrd_ori[i];
        rcParam.threshold_i[i] = value;
        rcParam.threshold_p[i] = value;
        rcParam.threshold_b[i] = value;
    }

    rcParam.direction = 8;
    rcParam.row_qp_delta = 0;

    ret = hi_mpi_venc_set_rc_param(ctx->channel_id, &rcParam);
    if (ret != HI_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_set_rc_param [%u] failed with ret 0x%x.\n", ctx->channel_id, ret);
        return ret;
    }

    return 0;
}

static av_cold int ff_himpi_enc_init(AVCodecContext *avctx)
{
    ASCENDEncContext_t *ctx = (ASCENDEncContext_t*)avctx->priv_data;
    AVASCENDDeviceContext *device_hwctx;
    AVHWFramesContext *hwframe_ctx;
    int ret = -1;
    int bitstream_buf_size = 0;
    char device_id[sizeof(int)];

    if (avctx == NULL || ctx == NULL) {
        av_log(avctx, AV_LOG_ERROR, "Early error in ff_himpi_enc_init.\n");
        return AVERROR_BUG;
    }

    if (ctx->hi_mpi_init_flag == 1) {
        av_log(avctx, AV_LOG_ERROR, "Error, himpi encoder double init.\n");
        return AVERROR_BUG;
    }

    if (encode_params_checking(avctx) != 0) {
        return AVERROR(EINVAL);
    }

    ctx->avctx = avctx;
    ctx->coded_width = FFALIGN(avctx->coded_width, VENC_WIDTH_ALIGN);
    ctx->coded_height = FFALIGN(avctx->coded_height, VENC_HEIGHT_ALIGN);
    sprintf(device_id, "%d", ctx->device_id);
    av_log(ctx, AV_LOG_INFO, "Device id is: %s.\n", device_id);

    if (avctx->pix_fmt == AV_PIX_FMT_ASCEND) {
        if (avctx->hw_frames_ctx) {
            av_buffer_unref(&ctx->hw_frame_ref);
            ctx->hw_frame_ref = av_buffer_ref(avctx->hw_frames_ctx);
            if (!ctx->hw_frame_ref) {
                return AVERROR(EINVAL);
            }
            hwframe_ctx = (AVHWFramesContext*)ctx->hw_frame_ref->data;
            ctx->hw_device_ref = av_buffer_ref(hwframe_ctx->device_ref);
            if (!ctx->hw_device_ref) {
                return AVERROR(EINVAL);
            }

            device_hwctx            = hwframe_ctx->device_ctx->hwctx;
            ctx->hw_frame_ctx       = hwframe_ctx;
            ctx->hw_device_ctx      = device_hwctx;
            ctx->ascend_ctx         = ctx->hw_device_ctx->ascend_ctx;
        } else {
            if (avctx->hw_device_ctx) {
                ctx->hw_device_ref = av_buffer_ref(avctx->hw_device_ctx);
                if (!ctx->hw_device_ref) {
                    return AVERROR(EINVAL);
                }
            } else {
                ret = av_hwdevice_ctx_create(&ctx->hw_device_ref, AV_HWDEVICE_TYPE_ASCEND, device_id, NULL, 0);
                if (ret < 0) {
                    return AVERROR(EINVAL);
                }
            }
            ctx->hw_frame_ref = av_hwframe_ctx_alloc(ctx->hw_device_ref);
            if (!ctx->hw_frame_ref) {
                av_log(avctx, AV_LOG_ERROR, "Failed in av_hwframe_ctx_alloc.\n");
                return AVERROR(EINVAL);
            }
            hwframe_ctx = (AVHWFramesContext*)ctx->hw_frame_ref->data;
            device_hwctx = hwframe_ctx->device_ctx->hwctx;
            ctx->hw_frame_ctx = hwframe_ctx;
            ctx->hw_device_ctx = device_hwctx;
            ctx->ascend_ctx = ctx->hw_device_ctx->ascend_ctx;

            if (!hwframe_ctx->pool) {
                hwframe_ctx->format = AV_PIX_FMT_ASCEND;
                hwframe_ctx->sw_format = avctx->sw_pix_fmt;
                hwframe_ctx->width = ctx->coded_width;
                hwframe_ctx->height = ctx->coded_height;
                if ((ret = av_hwframe_ctx_init(ctx->hw_frame_ref)) < 0) {
                    av_log(avctx, AV_LOG_ERROR, "Failed in av_hwframe_ctx_init.\n");
                    return AVERROR(EINVAL);
                }
            }
        }
        ctx->in_sw_pixfmt = avctx->sw_pix_fmt;
    } else {
        av_buffer_unref(&ctx->hw_device_ref);
        ret = av_hwdevice_ctx_create(&ctx->hw_device_ref, AV_HWDEVICE_TYPE_ASCEND, device_id, NULL, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed in av_hwdevice_ctx_create.\n");
            return AVERROR(EINVAL);
        }
        ctx->hw_device_ctx = ((AVHWDeviceContext*)ctx->hw_device_ref->data)->hwctx;
        ctx->ascend_ctx = ctx->hw_device_ctx->ascend_ctx;
        ctx->in_sw_pixfmt = avctx->pix_fmt;
    }

    bitstream_buf_size = ctx->coded_width * ctx->coded_height;
    switch (ctx->in_sw_pixfmt) {
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            bitstream_buf_size += (bitstream_buf_size >> 1);
            break;
        
        default:
            break;
    }

    ctx->eos_post_flag = 0;
    ctx->eos_received = 0;
    ctx->codec_abort_flag = 0;
    sem_init(&(ctx->eos_sema), 0, 0);
    ff_mutex_init(&ctx->queue_mutex, NULL);
    ctx->frame_queue = av_fifo_alloc(1000 * sizeof(StreamInfo_t));
    if (!ctx->frame_queue) {
        sem_post(&(ctx->eos_sema));
        av_log(avctx, AV_LOG_FATAL, "Failed to alloc memory for async fifo.\n");
        return AVERROR(EINVAL);
    }
    ctx->dataptr_queue = av_fifo_alloc(1000 * sizeof(uint8_t*));
    if (!ctx->dataptr_queue) {
        sem_post(&(ctx->eos_sema));
        av_log(avctx, AV_LOG_FATAL, "Failed to alloc memory for async fifo.\n");
        return AVERROR(EINVAL);
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed at line(%d) in func(%s), ret is %d", __LINE__, __func__, ret);
        return ret;
    }

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_init failed, ret is %d.\n", ret);
        return ret;
    }

    ret = set_venc_mod_param(avctx);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call set_venc_mod_param failed, ret is %d.\n", ret);
        return ret;
    }

    ret = create_venc_channel(ctx);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call create_venc_channel failed, ret is %d.\n", ret);
        return ret;
    }

    ret = set_venc_rc_param(ctx);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call set_venc_rc_param failed, ret is %d.\n", ret);
        return ret;
    }

    // start receive stream
    ctx->hi_mpi_init_flag = 1;
    ctx->thread_run_flag = 1;
    ctx->encode_run_flag = 1;
    
    ret = pthread_create(&ctx->thread_id, NULL, venc_get_stream, (void*)ctx);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Create receive stream thread failed, ret is %d.\n", ret);
        return ret;
    }

    avctx->pkt_timebase.num = 1;
    avctx->pkt_timebase.den = 90000;
    if (!avctx->pkt_timebase.num || !avctx->pkt_timebase.den) {
        av_log(avctx, AV_LOG_ERROR, "Invalid pkt_timebase.\n");
    }

    av_log(avctx, AV_LOG_DEBUG, "Ascend encoder init successfully.\n");
    return 0;
}

static int hi_mpi_encode(ASCENDEncContext_t *ctx, const AVFrame *frame)
{
    av_log(ctx, AV_LOG_DEBUG, "Send frame size: %ux%u, pts:%ld, frame type:%d.\n",
           frame->width, frame->height, frame->pts, frame->pict_type);
    
    hi_venc_start_param recvParam;
    recvParam.recv_pic_num = -1;
    int ret = hi_mpi_venc_start_chn(ctx->channel_id, &recvParam);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_start_chn failed, ret is %d.\n", ret);
        return ret;
    }

    if (frame && frame->width && frame->height) {
        uint8_t* streamBuffer = NULL;
        uint32_t dataSize = frame->width * frame->height * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
        ret = hi_mpi_dvpp_malloc(ctx->device_id, &streamBuffer, dataSize);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_dvpp_malloc failed, ret is %d.\n", ret);
            return ret;
        }

        uint32_t offset = 0;
        for (int i = 0; i < FF_ARRAY_ELEMS(frame->data) && frame->data[i]; i++) {
            size_t dstBytes = frame->width * frame->height * (i ? 1.0 / 2 : 1);
            ret = aclrtMemcpy(streamBuffer + offset, dataSize, frame->data[i], dstBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != 0) {
                hi_mpi_dvpp_free(streamBuffer);
                av_log(ctx, AV_LOG_ERROR, "Ascend memory H2D(host: %ld, dev: %d) failed, ret is %d.\n",
                       dstBytes, dataSize, ret);
                return ret;
            }
            offset += dstBytes;
        }

        hi_video_frame_info himpi_frame;
        himpi_frame.mod_id = HI_ID_VENC;
        himpi_frame.v_frame.width = frame->width;
        himpi_frame.v_frame.height = frame->height;
        himpi_frame.v_frame.field = HI_VIDEO_FIELD_FRAME;
        himpi_frame.v_frame.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        himpi_frame.v_frame.video_format = HI_VIDEO_FORMAT_LINEAR;
        himpi_frame.v_frame.compress_mode = HI_COMPRESS_MODE_NONE;
        himpi_frame.v_frame.dynamic_range = HI_DYNAMIC_RANGE_SDR8;
        himpi_frame.v_frame.color_gamut = HI_COLOR_GAMUT_BT709;
        himpi_frame.v_frame.width_stride[0] = FFALIGN(frame->width, VENC_WIDTH_ALIGN);

        himpi_frame.v_frame.width_stride[1] = FFALIGN(frame->width, VENC_WIDTH_ALIGN);
        himpi_frame.v_frame.height_stride[0] = FFALIGN(frame->height, VENC_HEIGHT_ALIGN);
        himpi_frame.v_frame.height_stride[1] = FFALIGN(frame->height, VENC_HEIGHT_ALIGN);
        himpi_frame.v_frame.virt_addr[0] = streamBuffer;
        himpi_frame.v_frame.virt_addr[1] = (hi_void *)((uintptr_t)himpi_frame.v_frame.virt_addr[0] +
                                                        frame->width * frame->height);
        himpi_frame.v_frame.frame_flag = 0;
        himpi_frame.v_frame.time_ref = 2 * ctx->frame_send_sum;
        himpi_frame.v_frame.pts = frame->pts;

        // push frame buffer addr to fifo
        ff_mutex_lock(&ctx->queue_mutex);
        av_fifo_generic_write(ctx->dataptr_queue, &streamBuffer, sizeof(uint8_t*), NULL);
        ff_mutex_unlock(&ctx->queue_mutex);

        ret = hi_mpi_venc_send_frame(ctx->channel_id, &himpi_frame, VENC_SEND_STREAM_TIMEOUT);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_send_stream failed, ret is %d.\n", ret);
            return ret;
        }

        ctx->frame_send_sum++;
    } else {
        if (!ctx->encoder_flushing) {
            ctx->encoder_flushing = 1;
        }
    }

    return 0;
}

static int hi_mpi_get_pkt(AVCodecContext *avctx, AVPacket *avpkt)
{
    ASCENDEncContext_t *ctx = (ASCENDEncContext_t*)avctx->priv_data;
    int ret = 0;
    if (!ctx->frame_queue || !ctx->dataptr_queue) {
        return AVERROR(EAGAIN);
    }

    StreamInfo_t stream_info;
    ff_mutex_lock(&ctx->queue_mutex);
    if (av_fifo_size(ctx->frame_queue) != 0) {
        av_fifo_generic_read(ctx->frame_queue, &stream_info, sizeof(stream_info), NULL);
    } else {
        ff_mutex_unlock(&ctx->queue_mutex);
        return AVERROR(EAGAIN);
    }
    ff_mutex_unlock(&ctx->queue_mutex);
    if (stream_info.event_type == EVENT_EOS) {
        return AVERROR_EOF;
    }

    ret = ff_get_encode_buffer(avctx, avpkt, stream_info.data_size, stream_info.data_size);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "FFmpeg get frame buffer failed, ret is %d.\n", ret);
        return AVERROR(EINVAL);
    }

    ret = aclrtMemcpy(avpkt->data, avpkt->size, stream_info.data, stream_info.data_size,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        hi_mpi_dvpp_free(stream_info.data);
        av_log(ctx, AV_LOG_ERROR, "Ascend memory D2H(dev: %d, host: %d) failed, ret is %d.\n",
               avpkt->size, stream_info.data_size, ret);
        return ret;
    }

    avpkt->pts = stream_info.pts;
    avpkt->size = stream_info.data_size;

    hi_mpi_dvpp_free(stream_info.data);

    return 0;
}

static int ff_himpi_enc_receive_packet(AVCodecContext *avctx, AVPacket *avpkt)
{
    int ret = -1;
    int send_ret = -1;
    int get_ret = -1;
    ASCENDEncContext_t *ctx = (ASCENDEncContext_t*)avctx->priv_data;
    AVFrame frame = {0};

    if (avctx == NULL || ctx == NULL || avpkt == NULL || !ctx->hi_mpi_init_flag) {
        av_log(avctx, AV_LOG_ERROR, "Early error in func: ff_himpi_enc_receive_packet");
        return AVERROR_EXTERNAL;
    }

    if (ctx->eos_received) {
        return AVERROR_EOF;
    }

    if (ctx->codec_abort_flag) {
        av_log(avctx, AV_LOG_FATAL, "Encoder got abort flag before send frame.\n");
        return AVERROR_EXTERNAL;
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed at line(%d) in func(%s), ret is %d", __LINE__, __func__, ret);
        return ret;
    }

    while(ctx->encode_run_flag) {
        if (!ctx->encoder_flushing) {
            send_ret = ff_encode_get_frame(avctx, &frame);
            if (send_ret < 0 && send_ret != AVERROR_EOF) {
                return send_ret;
            } else if (send_ret == AVERROR_EOF) {
                ctx->eos_post_flag = 1;
                ctx->encoder_flushing = 1;
                continue;
            }

            send_ret = hi_mpi_encode(ctx, &frame);
            av_frame_unref(&frame);
            if (send_ret < 0 && send_ret != AVERROR_EOF) {
                av_log(avctx, AV_LOG_ERROR, "Encoder send frame failed, ret is %d.\n", send_ret);
                return send_ret;
            }
        }

        get_ret = hi_mpi_get_pkt(avctx, avpkt);
        if (get_ret != 0 && get_ret != AVERROR_EOF) {
            if (get_ret != AVERROR(EAGAIN)) {
                return get_ret;
            }
            if (ctx->encoder_flushing) {
                av_usleep(2000);
            }
        } else {
            if (get_ret == AVERROR_EOF) {
                ctx->eos_received = 1;
            }
            return get_ret;
        }
    }
    av_log(avctx, AV_LOG_ERROR, "Encode stop, error.\n");
    return 0;
}

static av_cold int ff_himpi_enc_close(AVCodecContext *avctx)
{
    ASCENDEncContext_t *ctx = (ASCENDEncContext_t*)avctx->priv_data;
    int ret = -1;
    int semvalue = -1;
    struct timespec ts;

    if (avctx == NULL || ctx == NULL) {
        av_log(NULL, AV_LOG_ERROR, "Early error in ff_himpi_enc_close.\n");
        return AVERROR(EINVAL);
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed at line(%d) in func(%s), ret is %d", __LINE__, __func__, ret);
        return ret;
    }

    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 3;
    if (sem_timedwait(&(ctx->eos_sema), &ts) == -1) {
        sem_getvalue(&ctx->eos_sema, &semvalue);
        av_log(ctx, AV_LOG_ERROR, "Enc sem_timewait = -1, time out, semvalue = %d ...\n", semvalue);
    }

    if (ctx->hi_mpi_init_flag) {
        ret = hi_mpi_venc_stop_chn(ctx->channel_id);
        if (ret == HI_ERR_VENC_UNEXIST) {
            av_log(ctx, AV_LOG_WARNING, "Venc channel not exist, no need to stop it.\n", ret);
        } else if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_stop_chn failed, ret is %d.\n", ret);
            return ret;
        }

        ret = hi_mpi_venc_destroy_chn(ctx->channel_id);
        if (ret == HI_ERR_VENC_UNEXIST) {
            av_log(ctx, AV_LOG_WARNING, "Venc channel not exist, no need to destroy it.\n", ret);
        } else if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_venc_destroy_chn failed, ret is %d.\n", ret);
            return ret;
        }

        ret = hi_mpi_sys_exit();
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Call hi_mpi_sys_exit failed, ret is %d.\n", ret);
            return ret;
        }

        ctx->hi_mpi_init_flag = 0;
    }

    if (ctx->thread_run_flag) {
        ctx->thread_run_flag = 0;
        pthread_join(ctx->thread_id, NULL);
    }

    ctx->encode_run_flag = 0;
    
    sem_destroy(&(ctx->eos_sema));

    ff_mutex_lock(&ctx->queue_mutex);
    if (ctx->frame_queue) {
        av_fifo_freep(&ctx->frame_queue);
        ctx->frame_queue = NULL;
    }
    if (ctx->dataptr_queue) {
        av_fifo_freep(&ctx->dataptr_queue);
        ctx->dataptr_queue = NULL;
    }

    ff_mutex_unlock(&ctx->queue_mutex);
    ff_mutex_destroy(&ctx->queue_mutex);

    if (avctx->extradata) {
        av_free(avctx->extradata);
        avctx->extradata = NULL;
    }

    if (ctx->hw_frame_ref) {
        av_buffer_unref(&ctx->hw_frame_ref);
    }

    if (ctx->hw_device_ref) {
        av_buffer_unref(&ctx->hw_device_ref);
    }

    av_log(avctx, AV_LOG_DEBUG, "Encode closed.\n");

    return 0;
}






#define OFFSET(x) offsetof(ASCENDEncContext_t, x)
#ifndef VE
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
#endif

static const AVOption options[] = {
    { "device_id",      "Use to choose the ascend chip.",                   OFFSET(device_id), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 8, VE},
    { "channel_id",     "Set channelId of encoder.",                        OFFSET(channel_id), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 127, VE},
    { "profile",        "0: baseline, 1:main, 2: high. H265 file only support main level.",  OFFSET(profile), AV_OPT_TYPE_INT, { .i64 = 1}, 0, 2, VE},
    { "rc_mode",        "0: CBR mode, 1: VBR mode.",                        OFFSET(rc_mode), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 1, VE},
    { "gop",            "Set gop of encoder.",                              OFFSET(gop), AV_OPT_TYPE_INT, { .i64 = 30}, 1, 65536, VE},
    { "frame_rate",     "Set input stream frame_rate.",                     OFFSET(frame_rate), AV_OPT_TYPE_INT, { .i64 = 25}, 1, 240, VE},
    { "max_bit_rate",   "Set max_bite_rate of VBR or average_bit_rate of CBR.", OFFSET(max_bit_rate), AV_OPT_TYPE_INT, { .i64 = 20000}, 2, 614400, VE},
    { "movement_scene", "0: static scene, 1: movement scene.",              OFFSET(is_movement_scene), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 1, VE},
    { NULL }
};

static const AVCodecHWConfigInternal* ascend_hw_configs[] = {
    &(const AVCodecHWConfigInternal) {
        .public = {
            .pix_fmt        = AV_PIX_FMT_ASCEND,
            .methods        = AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX | AV_CODEC_HW_CONFIG_METHOD_INTERNAL,
            .device_type    = AV_HWDEVICE_TYPE_ASCEND
        },
        .hwaccel = NULL,
    },
    NULL
};

#define ASCEND_ENC_CODEC(x, X) \
    static const AVClass x##_ascend_class = { \
        .class_name = #x "_ascend_enc", \
        .item_name = av_default_item_name, \
        .option = options, \
        .version = LIBAVUTIL_VERSION_INT, \
    }; \
    AVCodec ff_##x##_ascend_encoder = { \
        .name       = #x "_ascend", \
        .long_name  = NULL_IF_CONFIG_SMALL("Ascend HiMpi " #X " encoder"), \
        .type       = AVMEDIA_TYPE_VIDEO, \
        .id         = AV_CODEC_ID_##X, \
        .priv_data_size = sizeof(ASCENDEncContext_t), \
        .priv_class     = &x##_ascend_class, \
        .init           = ff_himpi_enc_init, \
        .close          = ff_himpi_enc_close, \
        .receive_packet = ff_himpi_enc_receive_packet, \
        .capabilities   = AV_CODEC_CAP_DELAY, \
        .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP, \
        .pix_fmts       = (const enum AVPixelFormat[]){ AV_PIX_FMT_NV12, \
                                                        AV_PIX_FMT_ASCEND, \
                                                        AV_PIX_FMT_NONE }, \
        .hw_configs     = ascend_hw_configs, \
        .wrapper_name   = "ascendenc", \
    };

#if CONFIG_H264_ASCEND_ENCODER
ASCEND_ENC_CODEC(h264, H264)
#endif
#if CONFIG_H265_ASCEND_ENCODER
ASCEND_ENC_CODEC(h265, H265)
#endif
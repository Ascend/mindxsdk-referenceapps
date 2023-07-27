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

#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <semaphore.h>
#include <stdatomic.h>
#include "ascend_dec.h"

static void *get_frame(void *arg)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)arg;
    int ret = 0;
    int eos_flag = 0;
    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed, ret is %d.\n", ret);
        return ((void*) (-1));
    }

    hi_video_frame_info frame;
    hi_vdec_stream stream;
    hi_vdec_supplement_info stSupplement;

    av_log(NULL, AV_LOG_INFO, "Thread start.\n");

    while (ctx->thread_run_flag) {
        ret = hi_mpi_vdec_get_frame(ctx->channel_id, &frame, &stSupplement, &stream, VDEC_GET_TIME_OUT);
        if (ret != 0) {
            if (ctx->decoder_flushing && ret == HI_ERR_VDEC_BUF_EMPTY) {
                eos_flag = 1;
                av_log(ctx, AV_LOG_DEBUG, "Decoder flushing or stream eos.\n");
            } else {
                av_log(ctx, AV_LOG_DEBUG, "HiMpi get frame failed, ret is %d.\n", ret);
                continue;
            }
        }

        size_t decResult = frame.v_frame.frame_flag;
        if (eos_flag) {
            // eos
            FrameInfo_t frame_info;
            memset(&frame_info, 0, sizeof(FrameInfo_t));
            frame_info.event_type = EVENT_EOS;

            ff_mutex_lock(&ctx->queue_mutex);
            av_fifo_generic_write(ctx->frame_queue, &frame_info, sizeof(FrameInfo_t), NULL);
            ff_mutex_unlock(&ctx->queue_mutex);
            sem_post(&ctx->eos_sema);
            av_log(ctx, AV_LOG_DEBUG, "Decode got eos.\n");
            break;
        }

        hi_mpi_dvpp_free(stream.addr);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi free stream failed, ret is %d.\n", ret);
        }
        
        if (decResult != 0 || frame.v_frame.virt_addr[0] == NULL || stream.need_display == HI_FALSE) {
            ret = hi_mpi_vdec_release_frame(ctx->channel_id, &frame);
            if (ret != 0) {
                av_log(ctx, AV_LOG_ERROR, "HiMpi release frame failed, ret is %d.", ret);
                return ((void*) (-1));
            }
            continue;
        }
        FrameInfo_t frame_info;
        frame_info.ascend_ctx = ctx;
        get_vdec_frame_info(&frame_info, frame);

        ff_mutex_lock(&ctx->queue_mutex);
        av_fifo_generic_read(ctx->dts_queue, &frame_info.dts, sizeof(int64_t), NULL);
        av_fifo_generic_write(ctx->frame_queue, &frame_info, sizeof(FrameInfo_t), NULL);
        ff_mutex_unlock(&ctx->queue_mutex);

        ret = hi_mpi_vdec_release_frame(ctx->channel_id, &frame);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi release frame failed, ret is %d.\n", ret);
            return ((void*) (-1));
        }

        ctx->total_out_frame_count++;
    }
    return NULL;
}

static inline int decode_params_checking(AVCodecContext* avctx)
{
    switch (avctx->codec->id) {
        case AV_CODEC_ID_H264:
        case AV_CODEC_ID_H265:
            if (avctx->width < 128 || avctx->height < 128 ||
                avctx->width > 4096 || avctx->height > 4096) {
                av_log(avctx, AV_LOG_ERROR,
                    "H264 decoder only support resolution: 128x128 ~ 4096x4096, now: %dx%d.\n",
                    avctx->width, avctx->height);
                return -1;
            }
            break;
    
        default:
            break;
    }

    return 0;
}

static av_cold int ff_himpi_decode_end(AVCodecContext *avctx)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)avctx->priv_data;
    int ret = 0;
    int semvalue = 0;
    struct timespec ts;
    if (ctx == NULL || avctx->priv_data == NULL) {
        av_log(ctx, AV_LOG_ERROR, "HiMpi decode end error, AVCodecContext is NULL.\n");
        return AVERROR_BUG;
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set Context failed, ret is %d.\n", ret);
        return ret;
    }

    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 3;
    if (sem_timedwait(&(ctx->eos_sema), &ts) == -1) {
        semvalue = -1;
        sem_getvalue(&ctx->eos_sema, &semvalue);
        av_log(ctx, AV_LOG_ERROR, "Decode sem_timewait = -1, semvalue = %d.\n", semvalue);
    }

    if (ctx->hi_mpi_init_flag) {
        ret = hi_mpi_vdec_stop_recv_stream(ctx->channel_id);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi stop receive stream failed, ret is %d.\n", ret);
            return ret;
        }

        ret = hi_mpi_vdec_destroy_chn(ctx->channel_id);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi destroy channel failed, ret is %d.\n", ret);
            return ret;
        }

        ret = hi_mpi_sys_exit();
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi sys exit failed, ret is %d.\n", ret);
            return ret;
        }
    }

    ctx->hi_mpi_init_flag = 0;
    ctx->decode_run_flag = 0;

    if (ctx->thread_run_flag) {
        ctx->thread_run_flag = 0;
        pthread_join(ctx->thread_id, NULL);
    }

    sem_destroy(&ctx->eos_sema);

    if (ctx->frame_queue) {
        av_fifo_freep(&ctx->frame_queue);
        ctx->frame_queue = NULL;
    }

    ff_mutex_destroy(&ctx->queue_mutex);
    if (ctx->bsf) {
        av_bsf_free(&ctx->bsf);
        ctx->bsf = NULL;
    }

    av_buffer_unref(&ctx->hw_frame_ref);
    av_buffer_unref(&ctx->hw_device_ref);

    av_log(avctx, AV_LOG_INFO, "Decode hw send packet count is: %llu.\n", ctx->total_out_frame_count);
    av_log(avctx, AV_LOG_INFO, "Decode hw out frame count is: %llu.\n", ctx->total_packet_count);

    return 0;
}

static int malloc_and_send_frame(AVCodecContext *avctx, const AVPacket *avpkt)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)avctx->priv_data;
    int ret = 0;
    if (ctx->first_packet) {
        if (avctx->extradata_size) {
            uint8_t* streamBuffer = NULL;
            ret = hi_mpi_dvpp_malloc(ctx->device_id, &streamBuffer, avctx->extradata_size);
            if (ret != 0) {
                av_log(avctx, AV_LOG_ERROR, "HiMpi malloc first packet failed, ret is %d.\n", ret);
                return ret;
            }
            ret = aclrtMemcpy(streamBuffer, avctx->extradata_size, avctx->extradata, avctx->extradata_size,
                              ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != 0) {
                av_log(avctx, AV_LOG_ERROR, "Mem copy H2D first packet failed, ret is %d.\n", ret);
                return ret;
            }

            hi_vdec_stream stream;
            stream.pts = avpkt->pts;
            stream.addr = streamBuffer;
            stream.len = avctx->extradata_size;
            stream.end_of_frame = HI_TRUE;
            stream.end_of_stream = HI_FALSE;
            stream.need_display = HI_FALSE;

            hi_vdec_pic_info pic_info;
            pic_info.vir_addr = 0;
            pic_info.buffer_size = 0;
            pic_info.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            ret = hi_mpi_vdec_send_stream(ctx->channel_id, &stream, &pic_info, VDEC_TIME_OUT);
            if (ret != 0) {
                av_log(avctx, AV_LOG_ERROR, "HiMpi vdec send first packet failed, ret is %d.\n", ret);
                return ret;
            }
        }
        ctx->first_packet = 0;
    }

    uint8_t* streamBuffer = NULL;
    ret = hi_mpi_dvpp_malloc(ctx->device_id, &streamBuffer, avpkt->size);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi malloc packet failed, ret is %d.\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(streamBuffer, avpkt->size, avpkt->data, avpkt->size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "Mem copy H2D first packet failed, ret is %d.\n", ret);
        return ret;
    }

    // create stream info
    hi_vdec_stream stream;
    stream.pts = avpkt->pts;
    stream.addr = streamBuffer;
    stream.len = avpkt->size;
    stream.end_of_frame = HI_TRUE;
    stream.end_of_stream = HI_FALSE;
    stream.need_display = HI_TRUE;

    ff_mutex_lock(&ctx->queue_mutex);
    av_fifo_generic_write(ctx->dts_queue, &avpkt->dts, sizeof(int64_t), NULL);
    ff_mutex_unlock(&ctx->queue_mutex);

    // create frame info
    hi_vdec_pic_info pic_info;
    pic_info.width = ctx->resize_width;         // Output image width,  supports resize, set 0 means no resize.
    pic_info.height = ctx->resize_height;        // Output image height, supports resize, set 0 means no resize.
    pic_info.width_stride = FFALIGN(ctx->vdec_width, VDEC_WIDTH_ALIGN);
    pic_info.height_stride = FFALIGN(ctx->vdec_height, VDEC_HEIGHT_ALIGN);
    uint32_t size = pic_info.width_stride * pic_info.height_stride * YUV_BGR_CONVERT_3 / YUV_BGR_CONVERT_2;
    if (ctx->resize_str && ctx->resize_width != 0 && ctx->resize_height != 0) {
        size = ctx->resize_width * ctx->resize_height * YUV_BGR_CONVERT_3 / YUV_BGR_CONVERT_2;
        pic_info.width_stride = FFALIGN(ctx->resize_width, VDEC_WIDTH_ALIGN);
        pic_info.height_stride = FFALIGN(ctx->resize_height, VDEC_HEIGHT_ALIGN);
    }
        
    pic_info.buffer_size = size;
    pic_info.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    void *picBuffer = NULL;
    
    ret = hi_mpi_dvpp_malloc(ctx->device_id, &picBuffer, size);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi malloc failed, ret is %d.\n", ret);
        return ret;
    }
    pic_info.vir_addr = (uint64_t)picBuffer;

    do {
        ret = hi_mpi_vdec_send_stream(ctx->channel_id, &stream, &pic_info, VDEC_TIME_OUT);
        if ((unsigned int)ret == HI_ERR_VDEC_BUF_FULL) {
            usleep(VDEC_SLEEP_TIME);
        }
    } while ((unsigned int)ret == HI_ERR_VDEC_BUF_FULL);

    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi send stream failed, ret is %d.\n", ret);
        return ret;
    }

    ctx->frame_id++;
    ctx->total_packet_count++;
    return 0;
}

static int hi_mpi_decode(AVCodecContext *avctx, const AVPacket *avpkt)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*) avctx->priv_data;
    int ret = 0;
    AVPacket packet = { 0 };
    AVPacket bsf_packet = { 0 };

    if (avpkt && avpkt->size && ctx->bsf) {
        ret = av_packet_ref(&packet, avpkt);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "av_packet_ref failed, ret(%d).\n", ret);
            return ret;
        }
        ret = av_bsf_send_packet(ctx->bsf, &packet);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "av_bsf_send_packet failed, ret(%d).\n", ret);
            av_packet_unref(&packet);
            return ret;
        }
        ret = av_bsf_receive_packet(ctx->bsf, &bsf_packet);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "av_bsf_receive_packet failed, ret(%d).\n", ret);
            return ret;
        }
        avpkt = &bsf_packet;
    }
    av_packet_unref(&packet);

    if (avpkt && avpkt->size) {
        ret = malloc_and_send_frame(avctx, avpkt);
        if (ret != 0) {
            av_packet_unref(avpkt);
            return AVERROR(EINVAL);
        }
    } else {
        if (!ctx->decoder_flushing) {
            hi_vdec_stream stream;
            stream.addr = NULL;
            stream.len = 0;
            stream.end_of_frame = HI_FALSE;
            stream.end_of_stream = HI_TRUE; // Stream end flag to flushing all data.
            stream.need_display = HI_TRUE;

            hi_vdec_pic_info pic_info;
            pic_info.vir_addr = 0;
            pic_info.buffer_size = 0;
            pic_info.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            ret = hi_mpi_vdec_send_stream(ctx->channel_id, &stream, &pic_info, -1);
            if (ret != 0) {
                av_packet_unref(avpkt);
                av_log(avctx, AV_LOG_ERROR, "Send last stream failed, ret is %d", ret);
                return ret;
            }
            ctx->decoder_flushing = 1;
        }
    }
    av_packet_unref(avpkt);
    return 0;
}

static int himpi_get_frame(AVCodecContext *avctx, AVFrame *frame)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)avctx->priv_data;
    int ret = 0;
    if (!ctx->frame_queue) {
        return AVERROR(EAGAIN);
    }

    FrameInfo_t frame_info;
    ff_mutex_lock(&ctx->queue_mutex);
    if (av_fifo_size(ctx->frame_queue) != 0) {
        av_fifo_generic_read(ctx->frame_queue, &frame_info, sizeof(FrameInfo_t), NULL);
    } else {
        ff_mutex_unlock(&ctx->queue_mutex);
        return AVERROR(EAGAIN);
    }
    ff_mutex_unlock(&ctx->queue_mutex);

    if (frame_info.event_type == EVENT_EOS) {
        return AVERROR_EOF;
    }

    if (avctx->pix_fmt == AV_PIX_FMT_ASCEND) {
        ret = av_hwframe_get_buffer(ctx->hw_frame_ref, frame, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "av_hwframe_get_buffer failed, ret is %d.\n", ret);
            return AVERROR(EINVAL);
        }
        ret = ff_decode_frame_props(avctx, frame);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "ff_decode_frame_props failed, ret is %d.\n", ret);
            return AVERROR(EINVAL);
        }
    } else {
        ret = ff_get_buffer(avctx, frame, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Decode ff_get_buffer failed, ret is %d.\n", ret);
            return AVERROR(EINVAL);
        }
    }

    frame->pkt_pos      = -1;
    frame->pkt_duration = 0;
    frame->pkt_size     = -1;
    frame->pts          = frame_info.pts;
    frame->pkt_pts      = frame->pts;
    frame->pkt_dts      = frame_info.dts;
    frame->width        = frame_info.width_stride;
    frame->height       = frame_info.height_stride;

    switch (frame_info.format) {
        case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
            if (avctx->pix_fmt == AV_PIX_FMT_ASCEND) {
                uint32_t offset = 0;
                for (int i = 0; i < 2; i++) {
                    size_t dstBytes = frame->width * frame->height * (i ? 1.0 / 2 : 1);
                    ret = aclrtMemcpy(frame->data[i], dstBytes, frame_info.data + offset, dstBytes,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE);
                    if (ret != 0) {
                        av_log(avctx, AV_LOG_ERROR, "Mem copy D2D failed, ret is %d.\n", ret);
                        hi_mpi_dvpp_free(frame_info.data);
                        return ret;
                    }
                    offset += dstBytes;
                }
            } else {
                uint32_t offset = 0;
                for (int i = 0; i < 2; i++) {
                    size_t dstBytes = frame->width * frame->height * (i ? 1.0 / 2 : 1);
                    ret = aclrtMemcpy(frame->data[i], dstBytes, frame_info.data + offset, dstBytes,
                                      ACL_MEMCPY_DEVICE_TO_HOST);
                    if (ret != 0) {
                        av_log(avctx, AV_LOG_ERROR, "Mem copy D2H failed, ret is %d.\n", ret);
                        hi_mpi_dvpp_free(frame_info.data);
                        return ret;
                    }
                    offset += dstBytes;
                }
            }
            ret = hi_mpi_dvpp_free(frame_info.data);
            if (ret != 0) {
                av_log(avctx, AV_LOG_ERROR, "HiMpi free data failed, ret is %d.\n", ret);
            }
            break;
        
        default:
            hi_mpi_dvpp_free(frame_info.data);
            av_log(avctx, AV_LOG_ERROR, "Unsupport pixfmt: %d.\n", (int)frame_info.format);
            break;
    }

    
    return 0;
}

static int ff_himpi_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)avctx->priv_data;
    AVPacket pkt = { 0 };
    int send_ret = -1;
    int get_ret = -1;
    int ret = 0;

    if (avctx == NULL || avctx->priv_data == NULL) {
        av_log(avctx, AV_LOG_ERROR, "ff_himpi_receive_frame error, AVCodecContext is NULL.\n");
        return AVERROR_BUG;
    }
    if (!ctx->hi_mpi_init_flag || !ctx->thread_run_flag || !ctx->decode_run_flag) {
        av_log(avctx, AV_LOG_ERROR, "ff_himpi_receive_frame error, AVCodecContext is NULL.\n");
        return AVERROR_BUG;
    }

    if (ctx->eos_received) {
        return AVERROR_EOF;
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Set context failed, ret is %d.\n", ret);
        return ret;
    }

    while (ctx->decode_run_flag) {
        if (!ctx->decoder_flushing) {
            send_ret = ff_decode_get_packet(avctx, &pkt);
            if (send_ret < 0 && send_ret != AVERROR_EOF) {
                return send_ret;
            }
            send_ret = hi_mpi_decode(avctx, &pkt);
            av_packet_unref(&pkt);
            if (send_ret < 0 && send_ret != AVERROR_EOF) {
                av_log(ctx, AV_LOG_ERROR, "Send packet failed, ret is %d.\n", send_ret);
                return send_ret;
            }
        }

        get_ret = himpi_get_frame(avctx, frame);
        if (get_ret != 0 && get_ret != AVERROR_EOF) {
            if (get_ret != AVERROR(EAGAIN)) {
                return get_ret;
            }
            if (ctx->decoder_flushing) {
                av_usleep(2000);
            }
        } else {
            if (get_ret == AVERROR_EOF) {
                ctx->eos_received = 1;
            }
            return get_ret;
        }
    }

    av_log(avctx, AV_LOG_ERROR, "Decode stop, error.\n");
    return AVERROR_BUG;

}

static av_cold int ff_himpi_decode_init(AVCodecContext *avctx)
{
    ASCENDContext_t *ctx = (ASCENDContext_t*)avctx->priv_data;
    AVASCENDDeviceContext *hw_device_ctx;
    AVHWFramesContext *hw_frame_ctx;
    const AVBitStreamFilter *bsf;
    int ret = 0;

    if (avctx == NULL || avctx->priv_data == NULL) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi decoder init failed, AVCodecContext is NULL.\n");
        return AVERROR_BUG;
    }

    if (ctx->hi_mpi_init_flag == 1) {
        av_log(avctx, AV_LOG_ERROR, "Error, himpi decode double init. \n");
        return AVERROR_BUG;
    }

    enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_ASCEND, AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
    avctx->pix_fmt = ff_get_format(avctx, pix_fmts);
    if (avctx->pix_fmt < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error, ff_get_format failed with format id: %d.\n", avctx->pix_fmt);
        return AVERROR_BUG;
    }

    ctx->avctx = avctx;

    if (ctx->resize_str) {
        ret = av_parse_video_size(&ctx->resize_width, &ctx->resize_height, ctx->resize_str);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Invalid resize param: %s, which should be {width}x{height}.\n",
                   ctx->resize_str);
            return AVERROR_BUG;
        }

        if (ctx->resize_width != FFALIGN(ctx->resize_width, VDEC_WIDTH_ALIGN) ||
            ctx->resize_height != FFALIGN(ctx->resize_height, VDEC_HEIGHT_ALIGN)) {
            av_log(avctx, AV_LOG_ERROR, "Invalid resize param: %s, which should be stride by %d and %d.\n",
                   ctx->resize_str, VDEC_WIDTH_ALIGN, VDEC_HEIGHT_ALIGN);
            return AVERROR_BUG;
        }

        if (ctx->resize_width < 128 || ctx->resize_height < 128 ||
            ctx->resize_width > 4096 || ctx->resize_height > 4096) {
                av_log(avctx, AV_LOG_ERROR, "Invalid resize param: %s, which should be in [128x128 ~ 4096x4096].\n",
                       ctx->resize_str, VDEC_WIDTH_ALIGN, VDEC_HEIGHT_ALIGN);
                return AVERROR_BUG; 
            }
        avctx->coded_width = ctx->resize_width;
        avctx->coded_height = ctx->resize_height;
    }

    if (!ctx->resize_str || (ctx->resize_height == avctx->height && ctx->resize_width == avctx->width)) {
        ctx->vdec_width = FFALIGN(avctx->width, VDEC_WIDTH_ALIGN);
        ctx->vdec_height = FFALIGN(avctx->height, VDEC_HEIGHT_ALIGN);
        ctx->resize_width = ctx->resize_height = 0;
    } else {
        av_log(avctx, AV_LOG_INFO, "Vdec resize: %dx%d.\n", ctx->resize_width, ctx->resize_height);
    }

    ctx->vdec_width = avctx->width;
    ctx->vdec_height = avctx->height;
    
    if (decode_params_checking(avctx) != 0) {
        return AVERROR(EINVAL);
    }
    av_log(avctx, AV_LOG_DEBUG, "Vdec width: %d.\n", ctx->vdec_width);
    av_log(avctx, AV_LOG_DEBUG, "Vdec height: %d.\n", ctx->vdec_height);

    if (avctx->hw_frames_ctx) {
        av_buffer_unref(&ctx->hw_frame_ref);
        ctx->hw_frame_ref = av_buffer_ref(avctx->hw_frames_ctx);
        if (!ctx->hw_frame_ref) {
            ret = AVERROR(EINVAL);
            goto error;
        }

        hw_frame_ctx = (AVHWFramesContext*)ctx->hw_frame_ref->data;
        if (!hw_frame_ctx->pool ||
            (ctx->vdec_width != hw_frame_ctx->width && ctx->resize_width != hw_frame_ctx->width)) {
            if (hw_frame_ctx->pool) {
                av_buffer_pool_uninit(&hw_frame_ctx->pool);
            }
            hw_frame_ctx->width     = ctx->resize_width == 0 ? ctx->vdec_width : ctx->resize_width;
            hw_frame_ctx->height    = ctx->resize_height == 0 ? ctx->vdec_height : ctx->resize_height;
            hw_frame_ctx->initial_pool_size = 2;
            hw_frame_ctx->format    = AV_PIX_FMT_ASCEND;
            hw_frame_ctx->sw_format = avctx->sw_pix_fmt;

            ret = av_hwframe_ctx_init(ctx->hw_frame_ref);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "HWFrame contex init failed.\n");
                return AVERROR(ENAVAIL);
            }
        }
        ctx->hw_device_ref = av_buffer_ref(hw_frame_ctx->device_ref);
        if (!ctx->hw_device_ref) {
            av_log(avctx, AV_LOG_ERROR, "Get hw_device_ref failed.\n");
            ret = AVERROR(EINVAL);
            goto error;
        }
    } else {
        if (avctx->hw_device_ctx) {
            ctx->hw_device_ref = av_buffer_ref(avctx->hw_device_ctx);
            if (!ctx->hw_device_ref) {
                av_log(avctx, AV_LOG_ERROR, "ref hwdevice failed.\n");
                ret = AVERROR(EINVAL);
                goto error;
            }
        } else {
            char dev_idx[sizeof(int)];
            sprintf(dev_idx, "%d", ctx->device_id);
            av_log(avctx, AV_LOG_INFO, "dev_idx: %s.\n", dev_idx);
            ret = av_hwdevice_ctx_create(&ctx->hw_device_ref, AV_HWDEVICE_TYPE_ASCEND, dev_idx, NULL, 0);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "hwdevice contex create failed.\n");
                goto error;
            }
        }
        ctx->hw_frame_ref = av_hwframe_ctx_alloc(ctx->hw_device_ref);
        if (!ctx->hw_frame_ref) {
            av_log(avctx, AV_LOG_ERROR, "av_hwframe_ctx_alloc failed, ret is %d.\n", ret);
            ret = AVERROR(EINVAL);
            goto error;
        }
        hw_frame_ctx = (AVHWFramesContext*)ctx->hw_frame_ref->data;
        if (!hw_frame_ctx->pool) {
            hw_frame_ctx->width         = ctx->resize_width == 0 ? ctx->vdec_width : ctx->resize_width;
            hw_frame_ctx->height        = ctx->resize_height == 0 ? ctx->vdec_height : ctx->resize_height;
            hw_frame_ctx->initial_pool_size = 2;
            hw_frame_ctx->format        = AV_PIX_FMT_ASCEND;
            hw_frame_ctx->sw_format     = avctx->sw_pix_fmt;
            ret = av_hwframe_ctx_init(ctx->hw_frame_ref);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "hwframe ctx init error, ret is %d.\n", ret);
                return AVERROR(EINVAL);
            }
        }
    }
    hw_device_ctx = ((AVHWDeviceContext*)ctx->hw_device_ref->data)->hwctx;
    ctx->hw_device_ctx          = hw_device_ctx;
    ctx->hw_frames_ctx          = hw_frame_ctx;
    ctx->ascend_ctx             = ctx->hw_device_ctx->ascend_ctx;

    ctx->device_id              = ctx->ascend_ctx->device_id;
    ctx->frame_id               = 0;
    ctx->eos_received           = 0;
    ctx->total_out_frame_count  = 0;
    ctx->total_packet_count     = 0;
    ctx->decoder_flushing       = 0;
    ctx->first_packet           = 1;

    ff_mutex_init(&ctx->queue_mutex, NULL);

    switch (avctx->codec->id)
    {
        case AV_CODEC_ID_H264:
            ctx->codec_type = HI_PT_H264;
            break;
        case AV_CODEC_ID_H265:
            ctx->codec_type = HI_PT_H265;
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Invalid codec type, %d.\n", avctx->codec->id);
            return AVERROR_BUG;
    }
    ctx->bsf = NULL;
    if (avctx->codec->id == AV_CODEC_ID_H264 || avctx->codec->id == AV_CODEC_ID_H265) {
        if (avctx->codec->id == AV_CODEC_ID_H264)
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
        else if (avctx->codec->id == AV_CODEC_ID_H265)
            bsf = av_bsf_get_by_name("hevc_mp4toannexb");
        if (!bsf) {
            ret = AVERROR_BSF_NOT_FOUND;
            goto error;
        }
        ret = av_bsf_alloc(bsf, &ctx->bsf);
        if (ret < 0)
            goto error;
        
        ret = avcodec_parameters_from_context(ctx->bsf->par_in, avctx);
        if (ret < 0) {
            av_bsf_free(&ctx->bsf);
            goto error;
        }
        ret = av_bsf_init(ctx->bsf);
        if (ret < 0) {
            av_bsf_free(&ctx->bsf);
            goto error;
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Invalid codec id, %d.\n", avctx->codec->id);
        return AVERROR_BUG;
    }

    ctx->frame_queue = av_fifo_alloc(1000 * sizeof(FrameInfo_t));
    if (!ctx->frame_queue) {
        av_log(avctx, AV_LOG_ERROR, "Failed to alloc frame fifo queue.\n");
        goto error;
    }

    ctx->dts_queue = av_fifo_alloc(1000 * sizeof(int64_t));
    if (!ctx->dts_queue) {
        av_log(avctx, AV_LOG_ERROR, "Failed to alloc dts fifo queue.\n");
        goto error;
    }

    ret = aclrtSetCurrentContext(ctx->ascend_ctx->context);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "Set context failed, ret is %d.\n", ret);
        goto error;
    }

    ret = hi_mpi_sys_init();
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi sys init failed, ret is %d.\n", ret);
        goto error;
    }

    ctx->chn_attr_.type         = ctx->codec_type;
    ctx->chn_attr_.mode         = HI_VDEC_SEND_MODE_FRAME;
    ctx->chn_attr_.pic_width    = ctx->vdec_width;
    ctx->chn_attr_.pic_height   = ctx->vdec_height;

    // Stream buffer size, Recommended value is width * height * 3 / 2
    ctx->chn_attr_.stream_buf_size = ctx->vdec_width * ctx->vdec_height * YUV_BGR_CONVERT_3 / YUV_BGR_CONVERT_2;
    ctx->chn_attr_.frame_buf_cnt = REF_FRAME_NUM + DISPLAY_FRAME_NUM + 1;

    // Create buf attribute
    ctx->buf_attr_.width        = ctx->chn_attr_.pic_width;
    ctx->buf_attr_.height       = ctx->chn_attr_.pic_height;
    ctx->buf_attr_.align        = 0;
    ctx->buf_attr_.bit_width    = HI_DATA_BIT_WIDTH_8;
    ctx->buf_attr_.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ctx->buf_attr_.compress_mode = HI_COMPRESS_MODE_NONE;

    ctx->chn_attr_.frame_buf_size = hi_vdec_get_pic_buf_size(ctx->chn_attr_.type, &ctx->buf_attr_);

    // Configure video decoder channel attribute
    ctx->chn_attr_.video_attr.ref_frame_num     = REF_FRAME_NUM;
    ctx->chn_attr_.video_attr.temporal_mvp_en   = HI_TRUE;
    ctx->chn_attr_.video_attr.tmv_buf_size      = hi_vdec_get_tmv_buf_size(ctx->chn_attr_.type,
                                                                           ctx->chn_attr_.pic_width,
                                                                           ctx->chn_attr_.pic_height);

    av_log(avctx, AV_LOG_INFO, "Channel Id is: %d.\n", ctx->channel_id);
    ret = hi_mpi_vdec_create_chn(ctx->channel_id, &ctx->chn_attr_);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi create vdec channel failed, ret is %d.\n", ret);
        goto error;
    }

    // reset channel param.
    ret = hi_mpi_vdec_get_chn_param(ctx->channel_id, &ctx->chn_param_);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi vdec get channel param failed, ret is %d.\n", ret);
        goto error;
    }

    ctx->chn_param_.video_param.dec_mode        = HI_VIDEO_DEC_MODE_IPB;
    ctx->chn_param_.video_param.compress_mode   = HI_COMPRESS_MODE_HFBC;
    ctx->chn_param_.video_param.video_format    = HI_VIDEO_FORMAT_TILE_64x16;
    ctx->chn_param_.display_frame_num           = DISPLAY_FRAME_NUM;
    ctx->chn_param_.video_param.out_order       = HI_VIDEO_OUT_ORDER_DISPLAY;

    ret = hi_mpi_vdec_set_chn_param(ctx->channel_id, &ctx->chn_param_);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi vdec set channel param failed, ret is %d.\n", ret);
        goto error;
    }

    ret = hi_mpi_vdec_start_recv_stream(ctx->channel_id);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "HiMpi vdec start receive stream failed, ret is %d.\n", ret);
        goto error;
    }
    ctx->hi_mpi_init_flag = 1;
    ctx->decode_run_flag = 1;

    // create callback thread
    ctx->thread_run_flag = 1;
    ret = pthread_create(&ctx->thread_id, NULL, get_frame, (void *)ctx);
    if (ret != 0) {
        av_log(avctx, AV_LOG_ERROR, "pthread_create callback thread failed, ret is %d.\n", ret);
        goto error;
    }

    avctx->pkt_timebase.num = 1;
    avctx->pkt_timebase.den = 90000;
    if (!avctx->pkt_timebase.num || !avctx->pkt_timebase.den) {
        av_log(avctx, AV_LOG_ERROR, "Invalid pkt_timebase.\n");
    }
    
    sem_init(&ctx->eos_sema, 0, 0);
    return 0;

error:
    sem_post(&ctx->eos_sema);
    ff_himpi_decode_end(avctx);
    return ret;
}

static void ff_himpi_flush(AVCodecContext *avctx) {
    ff_himpi_decode_end(avctx);
    ff_himpi_decode_init(avctx);
}

#define OFFSET(x) offsetof(ASCENDContext_t, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption options[] = {
    { "device_id",      "Use to choose the ascend chip.",                   OFFSET(device_id), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 8, VD},
    { "channel_id",     "Set channelId of decoder.",                        OFFSET(channel_id), AV_OPT_TYPE_INT, { .i64 = 0}, 0, 255, VD},
    { "resize",         "Resize (width)x(height).",                         OFFSET(resize_str), AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, VD},
    { NULL }
};

static const AVCodecHWConfigInternal* ascend_hw_configs[] = {
    &(const AVCodecHWConfigInternal) {
        .public = {
            .pix_fmt        = AV_PIX_FMT_ASCEND,
            .methods        = AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX | AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX | \
                              AV_CODEC_HW_CONFIG_METHOD_INTERNAL,
            .device_type    = AV_HWDEVICE_TYPE_ASCEND
        },
        .hwaccel = NULL,
    },
    NULL
};

#define ASCEND_DEC_CODEC(x, X) \
    static const AVClass x##_ascend_class = { \
        .class_name = #x "_ascend_dec", \
        .item_name = av_default_item_name, \
        .option = options, \
        .version = LIBAVUTIL_VERSION_INT, \
    }; \
    AVCodec ff_##x##_ascend_decoder = { \
        .name       = #x "_ascend", \
        .long_name  = NULL_IF_CONFIG_SMALL("Ascend HiMpi " #X " decoder"), \
        .type       = AVMEDIA_TYPE_VIDEO, \
        .id         = AV_CODEC_ID_##X, \
        .priv_data_size = sizeof(ASCENDContext_t), \
        .priv_class     = &x##_ascend_class, \
        .init           = ff_himpi_decode_init, \
        .close          = ff_himpi_decode_end, \
        .receive_frame  = ff_himpi_receive_frame, \
        .flush          = ff_himpi_flush, \
        .capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_AVOID_PROBING | AV_CODEC_CAP_HARDWARE, \
        .pix_fmts       = (const enum AVPixelFormat[]){ AV_PIX_FMT_ASCEND, \
                                                        AV_PIX_FMT_NV12, \
                                                        AV_PIX_FMT_NONE }, \
        .hw_configs     = ascend_hw_configs, \
        .wrapper_name   = "ascenddec", \
    };

#if CONFIG_H264_ASCEND_DECODER
ASCEND_DEC_CODEC(h264, H264)
#endif

#if CONFIG_H265_ASCEND_DECODER
ASCEND_DEC_CODEC(h265, H265)
#endif
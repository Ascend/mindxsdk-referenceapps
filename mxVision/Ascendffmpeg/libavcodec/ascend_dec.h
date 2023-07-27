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

#ifndef FFMPEG_ASCEND_ASCEND_DEC_H
#define FFMPEG_ASCEND_ASCEND_DEC_H

#include "libavutil/parseutils.h"
#include "libavutil/buffer.h"
#include "libavutil/mathematics.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_ascend.h"
#include "libavutil/fifo.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "libavutil/common.h"
#include "libavutil/pixdesc.h"
#include "libavutil/thread.h"
#include "libavutil/version.h"
#include "config.h"
#include "avcodec.h"
#include "decode.h"
#include "hwaccels.h"
#include "hwconfig.h"
#include "internal.h"
#include "libavutil/avutil.h"

#include "acl/dvpp/hi_dvpp.h"

#define VDEC_WIDTH_ALIGN 16
#define VDEC_HEIGHT_ALIGN 2

#define REF_FRAME_NUM 8
#define DISPLAY_FRAME_NUM 2
#define VDEC_TIME_OUT 1000
#define VDEC_GET_TIME_OUT 100
#define VDEC_SLEEP_TIME 1000

#define YUV_BGR_CONVERT_3 3
#define YUV_BGR_CONVERT_2 2

typedef enum {
    EVENT_NEW_FRAME = 0,
    EVENT_EOS = 1,
} eventType_t;

typedef struct FrameInfo {
    void *ascend_ctx;
    eventType_t event_type;
    uint8_t *data;
    uint32_t data_size;
    hi_pixel_format format;
    int64_t pts;
    int64_t dts;
    uint32_t width_stride;
    uint32_t height_stride;
} FrameInfo_t;

typedef struct ASCENDContext {
    AVClass* av_class;
    int device_id;
    int channel_id;

    pthread_t thread_id;
    volatile int thread_run_flag;
    volatile int hi_mpi_init_flag;

    /*
    struct {
        int x;
        int y;
        int w;
        int h;
    } crop;
    struct {
        int width;
        int height;
    } resize;
    */

    char *output_pixfmt;
    sem_t eos_sema;

    AVBufferRef *hw_device_ref;
    AVBufferRef *hw_frame_ref;
    AVBSFContext *bsf;
    AVCodecContext *avctx;
    AVFifoBuffer *frame_queue;
    AVFifoBuffer *dts_queue;
    AVASCENDDeviceContext *hw_device_ctx;
    AVHWFramesContext *hw_frames_ctx;
    AscendContext *ascend_ctx;

    hi_vdec_chn_attr chn_attr_;
    hi_pic_buf_attr buf_attr_;
    hi_vdec_chn_param chn_param_;

    AVMutex queue_mutex;

    int max_width;
    int max_height;
    int vdec_width;
    int vdec_height;
    int stride_align;
    char* resize_str;
    int resize_width;
    int resize_height;
    hi_payload_type codec_type;

    volatile int frame_id;
    int first_packet;
    volatile int first_seq;
    volatile int eos_received;
    volatile int decoder_flushing;
    volatile int decode_run_flag;
    unsigned long long total_packet_count;
    unsigned long long total_out_frame_count;

    hi_vdec_stream stream;
    hi_vdec_pic_info pic_info;
} ASCENDContext_t;

static inline void get_vdec_frame_info(FrameInfo_t* frame_info, hi_video_frame_info frame)
{
    uint32_t width_stride = frame.v_frame.width_stride[0];
    uint32_t height_stride = frame.v_frame.height_stride[0];
    frame_info->width_stride = width_stride;
    frame_info->height_stride = height_stride;
    frame_info->data_size = width_stride * height_stride * YUV_BGR_CONVERT_3 / YUV_BGR_CONVERT_2;
    frame_info->format = frame.v_frame.pixel_format;
    frame_info->pts = frame.v_frame.pts;
    frame_info->data = frame.v_frame.virt_addr[0];
}

#endif // FFMPEG_ASCEND_ASCEND_DEC_H
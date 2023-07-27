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


#ifndef FFMPEG_ASCEND_ASCEND_ENC_H
#define FFMPEG_ASCEND_ASCEND_ENC_H

#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include <libavutil/hwcontext_ascend.h>
#include <libavutil/hwcontext.h>
#include <libavutil/buffer.h>
#include <libavutil/mathematics.h>
#include <libavutil/fifo.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/common.h>
#include <libavutil/pixdesc.h>
#include <libavutil/thread.h>
#include <libavutil/version.h>
#include "config.h"
#include "avcodec.h"
#include "encode.h"
#include "hwaccels.h"
#include "hwconfig.h"
#include "internal.h"
#include "libavutil/avutil.h"

#include "acl/dvpp/hi_dvpp.h"

// send
#define VENC_WIDTH_ALIGN 16
#define VENC_HEIGHT_ALIGN 2
#define MAX_VENC_WIDTH 4096
#define MAX_VENC_HEIGHT 4096
#define YUV_BGR_SIZE_CONVERT_2 2
#define YUV_BGR_SIZE_CONVERT_3 3
#define BUF_SIZE_STRIDE 64
#define BUF_SIZE_TIMES 10
#define HI_VENC_CHN_ATTR_STATS_TIME 1
#define HI_VENC_H264_BASELINE_LEVEL 0
#define HI_VENC_H264_MAIN_LEVEL 1
#define HI_VENC_H264_HIGH_LEVEL 2
#define HI_VENC_H265_MAIN_LEVEL 0
#define HI_ODD_NUM_3 3
#define TIMESTAMP_QUEUE_SIZE 100
#define MAX_HIMPI_VENC_CHN_NUM 127
#define FIRST_FRAME_START_QP 32
#define THRESHOLD_OF_ENCODE_RATE_VECTOR_LEN 16
#define THRESHOLD_OF_ENCODE_RATE 255
#define VENC_SEND_STREAM_TIMEOUT 1000

//receive
#define HI_SYS_CREATE_EPOLL_SIZE 10
#define HI_MPI_SYS_WAIT_EPOLL_MAX_EVENTS 3
#define HI_DVPP_EPOLL_EVENT 1024
#define HI_DVPP_EPOLL_EVENT_NUM 1000
#define WAIT_TILL_TIMEOUT 1000
#define WAIT_GET_TILL_TIMEOUT 100
#define MAX_MEMORY_SIZE 2147483648
#define MAX_PACK_COUNT 100

typedef enum {
    EVENT_NEW_FRAME = 0,
    EVENT_EOS = 1,
} eventType_t;

typedef struct StreamInfo {
    void *ascend_ctx;
    eventType_t event_type;
    uint8_t* data;
    uint32_t data_size;
    int64_t pts;
    hi_venc_data_type data_type;
    uint32_t len;
    int is_frame_end;
} StreamInfo_t;

typedef struct ASCENDEncContext {
    AVClass                         *av_class;
    AVCodecContext                  *avctx;
    int                             device_id;
    int                             channel_id;
    pthread_t                       thread_id;
    int                             encoder_flushing;
    int                             codec_abort_flag;
    volatile int                    thread_run_flag;
    volatile int                    encode_run_flag;
    volatile int                    eos_post_flag;
    volatile int                    eos_received;
    volatile int                    hi_mpi_init_flag;
    volatile int                    frame_send_sum;

    hi_venc_chn_attr                chn_attr_;
    int                             profile;
    int                             rc_mode;
    int                             gop;
    int                             frame_rate;
    int                             max_bit_rate;
    int                             is_movement_scene;

    int                             coded_width;
    int                             coded_height;
    sem_t                           eos_sema;

    AVFifoBuffer                    *frame_queue;
    AVFifoBuffer                    *dataptr_queue;
    AVMutex                         queue_mutex;

    AVBufferRef                     *hw_frame_ref;
    AVBufferRef                     *hw_device_ref;

    AscendContext                   *ascend_ctx;
    AVASCENDDeviceContext           *hw_device_ctx;
    AVHWFramesContext               *hw_frame_ctx;
    enum AVPixelFormat              in_sw_pixfmt;

} ASCENDEncContext_t;

static inline void get_venc_stream_info(StreamInfo_t *stream_info, hi_venc_stream stream)
{
    stream_info->data         = stream.pack[0].addr + stream.pack[0].offset;
    stream_info->pts          = stream.pack[0].pts;
    stream_info->len          = stream.pack[0].len;
    stream_info->is_frame_end = stream.pack[0].is_frame_end;
    stream_info->data_type    = stream.pack[0].data_type;
    stream_info->data_size    = stream.pack[0].len - stream.pack[0].offset;
}

#endif // FFMPEG_ASCEND_ASCEND_ENC_H
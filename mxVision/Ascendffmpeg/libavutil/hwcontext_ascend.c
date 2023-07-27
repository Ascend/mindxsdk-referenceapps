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

#include "buffer.h"
#include "common.h"
#include "hwcontext.h"
#include "hwcontext_ascend.h"
#include "hwcontext_internal.h"
#include "mem.h"
#include "pixdesc.h"
#include "imgutils.h"
#include "acl/acl.h"
#include "acl/dvpp/hi_dvpp.h"

static int init_flag = 0;

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_NV12,
};

#define ASCEND_FRAME_ALIGNMENT 1

typedef struct ASCENDFramesContext {
    int width;
    int height;
} ASCENDFramesContext;

static int ascend_device_init(AVHWDeviceContext *ctx)
{
    AVASCENDDeviceContext *hwctx = ctx->hwctx;
    if(!hwctx->ascend_ctx) {
        hwctx->ascend_ctx = av_mallocz(sizeof(*hwctx->ascend_ctx));
        if (!hwctx->ascend_ctx) {
            return AVERROR_UNKNOWN;
        }
    }
    return 0;
}

static void ascend_device_uninit(AVHWDeviceContext *device_ctx)
{
    AVASCENDDeviceContext *hwctx = device_ctx->hwctx;

    if (hwctx->ascend_ctx) {
        av_freep(&hwctx->ascend_ctx);
        hwctx->ascend_ctx = NULL;
    }
}

static int ascend_device_create(AVHWDeviceContext *device_ctx, const char *device, AVDictionary *opts, int flags)
{
    AVASCENDDeviceContext *hwctx = device_ctx->hwctx;
    AscendContext *ascend_ctx = NULL;

    int ret = 0;
    int device_idx = 0;
    if (device) {
        device_idx = strtol(device, NULL, 0);
    }
    av_log(device_ctx, AV_LOG_INFO, "device id is: %d.\n", device_idx);

    if (ascend_device_init(device_ctx) < 0)
        goto error;
    
    ascend_ctx = hwctx->ascend_ctx;
    ascend_ctx->device_id = device_idx;

    if (!init_flag) {
        ret = aclInit(NULL);
        if (ret != 0) {
            av_log(device_ctx, AV_LOG_ERROR, "InitDevices failed, ret = %d.\n", ret);
            goto error;
        }
        init_flag = 1;
    }
    

    ret = aclrtSetDevice(device_idx);
    if (ret != 0) {
        av_log(device_ctx, AV_LOG_ERROR, "SetDevice failed, ret = %d.\n", ret);
        goto error;
    }

    aclrtContext context;
    ret = aclrtCreateContext(&context, device_idx);
    if(ret != 0) {
        av_log(device_ctx, AV_LOG_ERROR, "CreateContext failed, ret = %d.\n", ret);
        goto error;
    }

    hwctx->ascend_ctx->context = context;
    return 0;
    error:
        ascend_device_uninit(device_ctx);
        return AVERROR_UNKNOWN;

}

static int ascend_frames_get_constraints(AVHWDeviceContext *ctx, const void *hwconfig,
                                         AVHWFramesConstraints *constraints)
{
    int i;
    constraints->valid_sw_formats = av_malloc_array(FF_ARRAY_ELEMS(supported_formats) + 1,
                                                    sizeof(*constraints->valid_sw_formats));
    if (!constraints->valid_sw_formats) {
        return AVERROR_EXTERNAL;
    }

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        constraints->valid_sw_formats[i] = supported_formats[i];
    }

    constraints->valid_sw_formats[FF_ARRAY_ELEMS(supported_formats)] = AV_PIX_FMT_NONE;

    constraints->valid_hw_formats = av_malloc_array(2, sizeof(*constraints->valid_hw_formats));
    if (!constraints->valid_sw_formats) {
        return AVERROR_EXTERNAL;
    }

    constraints->valid_hw_formats[0] = AV_PIX_FMT_ASCEND;
    constraints->valid_hw_formats[1] = AV_PIX_FMT_NONE;

    return 0;

}

static void ascend_buffer_free(void * opaque, uint8_t* data)
{
    AVHWFramesContext *ctx = opaque;
    if (data) {
        aclError ret = hi_mpi_dvpp_free(data);
        data = NULL;
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi free faile: dev addr %p \n", data);
        }
    }
}

static AVBufferRef *ascend_pool_alloc(void *opaque, int size)
{
    AVHWFramesContext *ctx = opaque;
    AVHWDeviceContext *device_ctx = ctx->device_ctx;
    AVASCENDDeviceContext *hwctx = device_ctx->hwctx;
    AscendContext *ascend_ctx = hwctx->ascend_ctx;

    AVBufferRef *buffer = NULL;
    void *data = NULL;

    aclError ret = hi_mpi_dvpp_malloc(ascend_ctx->device_id, (void **)(&data), size);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "HiMpi Malloc failed: dev addr %p, size %d.\n", data, size);
        return NULL;
    }
    buffer = av_buffer_create((uint8_t*)data, size, ascend_buffer_free, ctx, 0);
    if (!buffer) {
        ret = hi_mpi_dvpp_free(data);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "HiMpi Free failed with no buffer: dev addr %p.\n", data);
        }
    }
    return buffer;
}

static int ascend_frames_init(AVHWFramesContext *ctx)
{
    ASCENDFramesContext * priv = ctx->internal->priv;
    int i;
    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (av_get_pix_fmt_name(ctx->sw_format) == 
            av_get_pix_fmt_name(supported_formats[i])) {
            break;
        }
    }

    if (i == FF_ARRAY_ELEMS(supported_formats)) {
        av_log(ctx, AV_LOG_ERROR, "Pixel format '%s' is not supported.\n",
               av_get_pix_fmt_name(ctx->sw_format));
        return AVERROR_EXTERNAL;
    }

    av_pix_fmt_get_chroma_sub_sample(ctx->sw_format, &priv->width, &priv->height);

    if (!ctx->pool) {
        int size = av_image_get_buffer_size(ctx->sw_format, ctx->width,
                                            ctx->height, ASCEND_FRAME_ALIGNMENT);
        if (size < 0)
            return size;
        
        ctx->internal->pool_internal = av_buffer_pool_init2(size, ctx, ascend_pool_alloc, NULL);
        if (!ctx->internal->pool_internal) {
            av_log(ctx, AV_LOG_DEBUG, "internal pool init failed.\n");
            return AVERROR_EXTERNAL;
        }
    }

    return 0;
}

static int ascend_get_buffer(AVHWFramesContext *ctx, AVFrame *frame)
{
    frame->buf[0] = av_buffer_pool_get(ctx->pool);
    if (!frame->buf[0])
        return AVERROR_EXTERNAL;
    
    int ret = av_image_fill_arrays(frame->data, frame->linesize, frame->buf[0]->data,
                               ctx->sw_format, ctx->width, ctx->height, ASCEND_FRAME_ALIGNMENT);
    if (ret < 0)
        return ret;
    
    frame->format = AV_PIX_FMT_ASCEND;
    frame->width = ctx->width;
    frame->height = ctx->height;

    return 0;
}

static int ascend_transfer_get_formats(AVHWFramesContext *ctx, enum AVHWFrameTransferDirection dir,
                                       enum AVPixelFormat **formats)
{
    enum AVPixelFormat *fmts;

    fmts = av_malloc_array(2, sizeof(*fmts));
    if (!fmts)
        return AVERROR_EXTERNAL;
    
    fmts[0] = ctx->sw_format;
    fmts[1] = AV_PIX_FMT_NONE;

    *formats = fmts;
    return 0;
}

static int ascend_transfer_data_to(AVHWFramesContext *ctx, AVFrame *dst, const AVFrame *src)
{
    AVHWDeviceContext *device_ctx = ctx->device_ctx;
    AVASCENDDeviceContext *hwctx = ctx->hwctx;
    AscendContext *ascend_ctx = hwctx->ascend_ctx;

    int i;
    size_t dstBytes;
    size_t srcBytes;
    aclError ret;
    for (i = 0; i < FF_ARRAY_ELEMS(src->data) && src->data[i]; i++) {
        dstBytes = src->width * src->height * (i ? 1.0 / 2 : 1);
        srcBytes = src->width * src->height * (i ? 1.0 / 2 : 1);
        ret = aclrtMemcpy(dst->data[i], dstBytes, src->data[i], srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Mem copy h2d: host %p wigh %lu -> dev %p with %lu.\n",
                   src->data[i], srcBytes, dst->data[i], dstBytes);
            av_log(ctx, AV_LOG_ERROR, "ascendMemcoy H2D error occur, func: %s, line %d.\n",
                   __func__, __LINE__);
            return -1;
        }
    }

    return 0;
}

static int ascend_transfer_data_from(AVHWFramesContext *ctx, AVFrame *dst, const AVFrame *src)
{
    AVHWDeviceContext *device_ctx = ctx->device_ctx;
    AVASCENDDeviceContext *hwctx = ctx->hwctx;
    AscendContext *ascend_ctx = hwctx->ascend_ctx;

    int i;
    size_t dstBytes;
    size_t srcBytes;
    aclError ret;
    for (i = 0; i < FF_ARRAY_ELEMS(src->data) && src->data[i]; i++) {
        dstBytes = src->width * src->height * (i ? 1.0 / 2 : 1);
        srcBytes = src->width * src->height * (i ? 1.0 / 2 : 1);
        ret = aclrtMemcpy(dst->data[i], dstBytes, src->data[i], srcBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != 0) {
            av_log(ctx, AV_LOG_ERROR, "Mem copy d2h: dev %p wigh %lu -> host %p with %lu.\n",
                   src->data[i], srcBytes, dst->data[i], dstBytes);
            av_log(ctx, AV_LOG_ERROR, "ascendMemcoy D2H error occur, func: %s, line %d.\n",
                   __func__, __LINE__);
            return -1;
        }
    }

    return 0;
}

const HWContextType ff_hwcontext_type_ascend = {
    .type                   = AV_HWDEVICE_TYPE_ASCEND,
    .name                   = "ASCEND",
    
    .device_hwctx_size      = sizeof(AVASCENDDeviceContext),
    .frames_priv_size       = sizeof(ASCENDFramesContext),
    
    .device_create          = ascend_device_create,
    .device_init            = ascend_device_init,
    .device_uninit          = ascend_device_uninit,
    .frames_get_constraints = ascend_frames_get_constraints,
    .frames_init            = ascend_frames_init,
    .frames_get_buffer      = ascend_get_buffer,
    .transfer_get_formats   = ascend_transfer_get_formats,
    .transfer_data_to       = ascend_transfer_data_to,
    .transfer_data_from     = ascend_transfer_data_from,

    .pix_fmts               = (const enum AVPixelFormat[]) {AV_PIX_FMT_ASCEND, AV_PIX_FMT_NONE},
};
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

#ifndef FFMPEG_ASCEND_HWCONTEXT_ASCEND_H
#define FFMPEG_ASCEND_HWCONTEXT_ASCEND_H

#include "acl/acl.h"
#include "pixfmt.h"

typedef struct AscendContext {
    int device_id;
    aclrtContext context;
} AscendContext;

typedef struct AVASCENDDeviceContext {
    AscendContext *ascend_ctx;
} AVASCENDDeviceContext;

#endif //FFMPEG_ASCEND_HWCONTEXT_ASCEND_H
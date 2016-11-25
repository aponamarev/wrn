--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************
--[[opt dependencies:
depth - depth should be 6n+4
num_classes
widen_factor
dropout
imageSize - require in line 26 of models/utils (to test created model) with default value 32
]]--
--
local nn = require 'nn'
local utils = paths.dofile'utils.lua'
--
local M = {}
local WRND = torch.class('WRND', M)
function WRND:__init(builder)
  self._builder = builder
end
function WRND:construct()
  self._builder:create_net()
  self._builder:reset_input()
  --self._builder:disable_bias()
  self._builder:msr_init()
  self._builder:fc_init()
  self._builder:test_net()
end
function WRND:getWRN()
  if not self._builder._product.net then self:construct() end
  return self._builder._product.net
end
local WRNB = torch.class('WRNB', M)
function WRNB:__init(product)
  self._product = product
end
function WRNB:create_net()
  self._product.net = self._product:create()
end
function WRNB:test_net()
  utils.testModel(self._product.net, self._product.img_cropsize)
end
function WRNB:msr_init()
  utils.MSRinit(self._product.net)
end
function WRNB:fc_init()
  utils.FCinit(self._product.net)
end
function WRNB:disable_bias()
  utils.DisableBias(self._product.net)
end
function WRNB:reset_input()
  self._product.net:get(1).gradInput = nil
end
--
local WRN = torch.class('WRN', M)
function WRN:__init(opt)
  local optErrorMSG = "Error: Wide Residual Net (WRN) requires 4 parameters: depth, dropout, num_classes, and widen_factor to be stored in opt dictionary"
  assert(opt and opt.depth and opt.num_classes and opt.widen_factor, optErrorMSG)
  assert((opt.depth - 6) % 6 == 0, 'Error: depth parameter should be 6n+6')
  self.depth = opt.depth
  self.num_classes = opt.num_classes
  self.widen_factor = opt.widen_factor
  self.dropout = opt.dropout 
  self.img_cropsize = opt.imageSize
end
--
function WRN:create()
  local n = nn.Sequential()
  n:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(16))
  --n:add(nn.SpatialConvolution(16,16,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(16))
  --n:add(nn.SpatialConvolution(16,16,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(16))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --2
  --n:add(nn.SpatialConvolution(16,16,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(16))
  --n:add(nn.SpatialConvolution(16,16,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(16))
  n:add(nn.SpatialConvolution(16,32,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(32))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --4
  --n:add(nn.SpatialConvolution(32,32,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(32))
  --n:add(nn.SpatialConvolution(32,32,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(32))
  n:add(nn.SpatialConvolution(32,64,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(64))
  --n:add(nn.SpatialConvolution(64,64,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(64))
  --n:add(nn.SpatialConvolution(64,64,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(64))
  n:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(64))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --8
  --n:add(nn.SpatialConvolution(64,64,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(64))
  --n:add(nn.SpatialConvolution(64,64,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(64))
  n:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(128))
  --n:add(nn.SpatialConvolution(128,128,1,3,1,1,0,1)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(128))
  --n:add(nn.SpatialConvolution(128,128,3,1,1,1,1,0)):add(nn.ReLU(true))
  --n:add(nn.SpatialBatchNormalization(128))
  n:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(128))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --16
  n:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(256))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --32
  n:add(nn.View((self.img_cropsize/32)^2*256):setNumInputDims(3))
  n:add(nn.Dropout(0.2))
  n:add(nn.Linear((self.img_cropsize/32)^2*256,self.num_classes*2, true))
  n:add(nn.Linear(self.num_classes*2,self.num_classes*1.5, true))
  n:add(nn.Linear(self.num_classes*1.5,self.num_classes*1.25, true))
  n:add(nn.Linear(self.num_classes*1.25,self.num_classes,true))
  n = n:float()
  return n
end
--external interface
local eM = {}
local Interface = torch.class('Interface', eM)
function Interface:__init(opt)
  self.director = M.WRND(M.WRNB(M.WRN(opt)))
end
function Interface:get()
  local net = self.director:getWRN()
  self.director = nil
  return net
end
return eM.Interface




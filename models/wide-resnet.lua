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
  self._builder:disable_bias()
  self._builder:test_net()
  self._builder:msr_init()
  self._builder:fc_init()
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
  assert(opt and opt.depth and opt.num_classes and opt.widen_factor and opt.dropout, optErrorMSG)
  assert((opt.depth - 2) % 10 == 0, 'Error: depth parameter should be 6n+6')
  self.depth = opt.depth
  self.num_classes = opt.num_classes
  self.widen_factor = opt.widen_factor
  self.dropout = opt.dropout 
  self.img_cropsize = opt.imageSize
end
--
function WRN:create()
  --creating alliaces that actually make sence
  --local Convolution = nn.SpatialConvolutionMM
  local Convolution = nn.SpatialConvolution
  local Avg = nn.SpatialAveragePooling
  local ReLU = nn.ReLU
  local Max = nn.SpatialMaxPooling
  local SBatchNorm = nn.SpatialBatchNormalization
  local function Dropout()
    return nn.Dropout(self.dropout or 0,nil,true)
  end

  local depth = self.depth

  local blocks = {}

  local function wide_basic(nInputPlane, nOutputPlane, stride)
    --[[
    wide basic:
    Block - nn.Sequential
    a) nn.ConcatTable()
    b) Conv
      1. Spatial batch normalization
      2. ReLu
      3. Convolution
      4. Spatial batch normalization
      5. ReLu
      6. Dropout
      7. Spatial Convolution
    c) Shortcut - nn.Identity
    d) nn.CAddTable(true)
    ]]--
    local conv_params = {
      {3,3,stride,stride,1,1},
      {3,3,1,1,1,1},
    }
    local nBottleneckPlane = nOutputPlane
    local block = nn.Sequential()
    local convs = nn.Sequential()     
    for i,v in ipairs(conv_params) do
      if i == 1 then
        local module = nInputPlane == nOutputPlane and convs or block
        module:add(ReLU(true)):add(SBatchNorm(nInputPlane))
        convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
      else
        convs:add(ReLU(true)):add(SBatchNorm(nBottleneckPlane))
        if self.dropout > 0 then
          convs:add(Dropout())
        end
        convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
      end
    end
    local shortcut = nInputPlane == nOutputPlane and
    nn.Identity() or
    Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
    return block
    :add(nn.ConcatTable()
      :add(convs)
      :add(shortcut))
    :add(nn.CAddTable(true))
  end

  -- Stacking Residual Units on the same stage
  local function layer(block, nInputPlane, nOutputPlane, count, stride)
    local s = nn.Sequential()
--block is an aliace for wide_basic function from line 102
    --[[
    wide basic:
    Block - nn.Sequential
    a) nn.ConcatTable()
    b) Conv - nn.Sequential
      1. Spatial batch normalization
      2. ReLu
      3. Convolution (3x3, 1,1 1,1 - returns same 2d size)
      4. Spatial batch normalization
      5. ReLu
      6. Dropout
      7. Spatial Convolution (3x3, 1,1 1,1 - returns same 2d size)
    c) Shortcut - nn.Identity
    d) nn.CAddTable(true)
    ]]--
    s:add(block(nInputPlane, nOutputPlane, stride))
    for i=2,count do
      s:add(block(nOutputPlane, nOutputPlane, 1))
    end
    return s
  end

  local model = nn.Sequential()
  do
    assert((depth-2) % 10 == 0, 'Error: depth should be devisible by 12')
    local n = (depth-2) / 10

    local k = self.widen_factor
    local nStages = torch.Tensor{8, 8*k, 16*k, 32*k, 64*k, 128*k} --k is a whidenning factor

    model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 256x256)
    model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size - same: 256x256)
    model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- The last element adds stride 2 - similar to pulling - Stage 2 (spatial size - 2x smaller)
    model:add(ReLU(true)):add(SBatchNorm(nStages[3]))
    model:add(Avg(2,2, 2,2)) -- 4x smaller
    model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- 8x smaller
    model:add(layer(wide_basic, nStages[4], nStages[5], n, 2)) -- 16x smaller
    model:add(layer(wide_basic, nStages[5], nStages[6], n, 2)) -- 32x smaller
    model:add(ReLU(true)):add(SBatchNorm(nStages[6]))
    model:add(Avg(2,2, 2,2)) -- 64 smaller
    local numOfFeatures = nStages[6]*(self.img_cropsize/64)^2
    model:add(nn.View(numOfFeatures):setNumInputDims(3))
    model:add(nn.Linear(numOfFeatures, self.num_classes))
  end

  return model
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




--[[
Class DummyNet is designed to proved a testing sandbox for network procedures and test usability of other elements of the training process
Requirements:
1. Provide easy to use net interface to get net and remove all other enements from the memroy
2. Put together a net consisting of several blocks

Objects required
1. Interface Object
1.1 :__Init(netOptions) initiates an object and stores a director in self._director
1.2 :getNet() - returns net and cleans the self._director = nil

2 Director Object
2.1 :__Init(builder) - stores a builder to be assembled in a self._builder
2.2 :assemble() - lunches a sequence of methods to assemble the net

3 Builder Object
3.1 :__init(product) - stores a product to be assembles in self._product
3.2 :addConvBlock() - adds a convolution block to a self._product
3.3 addLinearBlock() - adds a linear block to self._product
]]--
--require('mobdebug').start()
require 'nn'
local M = {}
local InterfaceMetatable = {}
Interface = torch.class('Interface', InterfaceMetatable)

function Interface:__init(netOptions)
  self._director = M.Director(M.Builder(M.Product()), netOptions)
  self._director:assemble()
end
function Interface:getNet()
  local n = nn.Sequential()
  n:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(16))
  n:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(16))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --2
  n:add(nn.SpatialConvolution(16,32,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(32))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --4
  n:add(nn.SpatialConvolution(32,64,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(64))
  n:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(64))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --8
  n:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(128))
  n:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(128))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --16
  n:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1)):add(nn.ReLU(true))
  n:add(nn.SpatialBatchNormalization(256))
  n:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --32
  n:add(nn.View((32/32)^2*256):setNumInputDims(3))
  n:add(nn.Dropout(0.2))
  n:add(nn.Linear((32/32)^2*256,80, true))
  n:add(nn.Linear(80,40, true))
  n:add(nn.Linear(40,20, true))
  n:add(nn.Linear(20,10,true))
  n = n:float()
  return n
end

local Director = torch.class('Director', M)
function Director:__init(builder, options)
  self.opt = options
  self._builder = builder
end
function Director:assemble()
  self._builder:addConvBlock(3, 16)
  self._builder:addMaxPulling()
  self._builder:addConvBlock(16, 32)
  self._builder:addMaxPulling()
  self._builder:addConvBlock(32, 64)
  self._builder:addMaxPulling()
  self._builder:addConvBlock(64, 64)
  self._builder:addMaxPulling()
  self._builder:addConvBlock(64, 64)
  self._builder:addMaxPulling()
  --[[self._builder:addConvBlock(32, 64)
  self._builder:addMaxPulling()]]--
  self._builder:addLinearBlock(self.opt.dropout)
end

local Builder = torch.class('Builder', M)
function Builder:__init(product)
  self._product = product
end
function Builder:addConvBlock(inputSize, outputSize)
  self._product.net:add(nn.SpatialConvolution(inputSize,outputSize,3,3,1,1,1,1))
  self._product.net:add(nn.SpatialBatchNormalization(outputSize)):add(nn.ReLU(true))
  self._product.net:add(nn.SpatialConvolution(outputSize,outputSize,3,3,1,1,1,1))
  self._product.net:add(nn.SpatialBatchNormalization(outputSize)):add(nn.ReLU(true))
end
function Builder:addMaxPulling()
  self._product.net:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
end
function Builder:addLinearBlock(dropout)
  self._product.net:add(nn.View(-1):setNumInputDims(3))
  self._product.net:add(nn.Linear(4*4*64, 100)):add(nn.ReLU(true))
  self._product.net:add(nn.Linear(100, 13))
end
local Product = torch.class('Product', M)
function Product:__init()
  self.net = nn.Sequential()
end

return InterfaceMetatable.Interface


if not torch then require 'torch' end
--if not class then class = require 'class' end
local Scale = require 'DataProvider/CACD2000/Scale'
local Flip = require 'DataProvider/CACD2000/Flip'
local Crop = require 'DataProvider/CACD2000/Crop'
local FiveCrop = require 'DataProvider/CACD2000/FiveCrop'
local M = {}
local TransformBuilder = torch.class('TransformBuilder', M)
function TransformBuilder:__init(transform)
  self.Transform = transform
end
function TransformBuilder:scale(opt)
  self.Transform:add(Scale(opt))
end
function TransformBuilder:flip(opt)
  self.Transform:add(Flip(opt))
end
function TransformBuilder:crop(opt)
  self.Transform:add(Crop(opt))
end
function TransformBuilder:fivecrop(opt)
  self.Transform:add(FiveCrop(opt))
end
return M.TransformBuilder
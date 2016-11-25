if not torch then require 'torch' end
--if not class then class = require 'class' end
local Transformer = require 'DataProvider/CACD2000/Transformer'
local TransformBuilder = require 'DataProvider/CACD2000/TransformBuilder'
local M = {}
local TransformDirector = torch.class('TransformDirector', M)
function TransformDirector:__init(opt)
  assert(opt, "Error: opt was not provided to TransformaDirector class at initiation. Opt should contin opt.scale, opt.flip, opt.crop. opt.fivecrop")
  self.opt = opt
  self.container = TransformBuilder(Transformer())
end
function TransformDirector:assemble()
  if self.opt.scale then self.container:scale(self.opt.scale) end
  if self.opt.flip then self.container:flip(self.opt.flip.vertircal) end
  if self.opt.crop then self.container:crop(self.opt.crop) end
  if self.opt.fivecrop then self.container:fivecrop(self.opt.fivecrop) end
  return self.container.Transform
end
return M.TransformDirector
--[[local d = TransformDirector({scale = 320, flip = true, fivecrop = 280})
local t = d:assemble()
if not image then require 'image' end
local img = image.lena()
img = t:exec(img)
image.display(img)]]--
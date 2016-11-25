if not torch then require 'torch' end
--if not class then class = require 'class' end
local ImgLoader = require 'DataProvider/CACD2000/ImageLoader'
local TransformDirector = require 'DataProvider/CACD2000/TransformDirector'
--local Normalizer = require 'DataProvider/CACD2000/Normalizer'
local Whitening = require 'DataProvider/CACD2000/Whitening'
local M = {}
local ImgeGenerator = torch.class('ImgeGenerator', M)
function ImgeGenerator:__init(opt)
  assert(opt, "Error: opt was not provided to ImageGenerator class at initiation. Opt should contain opt.dataset_path, opt.scale, opt.flip, opt.crop, or opt.fivecrop")
  assert(opt.dataset_path, "Error: opt.dataset_path was not provided to ImageGenerator class at initiation.Provide a location of the training set here.")
  self.container = {}

  self.container[1] = ImgLoader(opt.dataset_path)
  if opt.mean_std_path then
    --table.insert(self.container, #self.container + 1,Normalizer(opt.mean_std_path))
    table.insert(self.container, #self.container + 1, Whitening())
  end
  table.insert(self.container, #self.container + 1,TransformDirector(opt):assemble())
  self.multiple = self.container[#self.container].multiple
end
function ImgeGenerator:exec(file_name)
  local img
  for k, v in pairs(self.container) do
    if k == 1 then
      img = v:exec(file_name)
    else
      img = v:exec(img)
    end
  end
  return img
end
return M.ImgeGenerator
--[[local opt = {
  dataset_path = '/Users/aponamaryov/Downloads/training/Aaron_Ashmore',
  scale = 320,
  flip = {vertircal = nil},
  fivecrop = 280
}
local imgGenerator = ImgeGenerator(opt)
local img = imgGenerator:exec('25_Aaron_Ashmore_0003.jpg')
print(#img)
require 'image'
image.display(img)
]]--
if not torch then require 'torch' end
if not paths then paths = require 'paths' end

local OnePath = require 'DataProvider/ImgFinderOnePath'
local TwoPaths = require 'DataProvider/ImgFinderTwoPath'

local M = {}
local ClassPathGenerator = torch.class("ClassPathGenerator", M)
function ClassPathGenerator:__init(opt, classToIdx)
  self.one_path = OnePath()
  self.two_paths = TwoPaths()
  self.train_path = opt.train_path
  self.val_path = opt.val_path
  self.val_sample = opt.val_sample or 0.05
  self.classToIdx = classToIdx
end
function ClassPathGenerator:train_path_only()
  local valImagePath, valImageClass = {}, {}
  local trainImagePath, trainImageClass, maxlength = self.one_path:launch(self.train_path, self.classToIdx)
  local n_files = #trainImagePath
  local sample_size = math.floor(n_files * self.val_sample)
  for i = 1, sample_size do
    local rnd = math.random(1,#trainImagePath)
    table.insert(valImagePath, table.remove(trainImagePath, rnd))
    table.insert(valImageClass, table.remove(trainImageClass, rnd))
  end
  local valImagePath, valImageClass = self.one_path:optimize(valImagePath, valImageClass, maxlength)
  local trainImagePath, trainImageClass = self.one_path:optimize(trainImagePath, trainImageClass, maxlength)
  return valImagePath, valImageClass, trainImagePath, trainImageClass
end
function ClassPathGenerator:train_val_paths()
  return self.two_paths:launch(self.train_path, self.val_path, self.classToIdx)
end
function ClassPathGenerator:launch()
  --check if paths and sample size are provided
  assert(self.train_path and paths.dirp(self.train_path), "Error: opt.train_path was not provided or wasn't correct")
  if not self.val_path then
    assert(self.val_sample, "Error: you have to either provide both train_path and val_path or train_path and val_sample")
    return self:train_path_only()
  else
    assert(paths.dirp(self.val_path), "Error: provided val_path was incorrect.")
    return self:train_val_paths()
  end
end
--[[
local class_path_generator = M.ClassPathGenerator({
    train_path = '/Users/aponamaryov/TorchTutorials/fb.resnet.torch/datasets/Data/train',
    val_path = '/Users/aponamaryov/TorchTutorials/fb.resnet.torch/datasets/Data/val',
    val_sample = 0.1})
print(class_path_generator:launch())]]--
return M.ClassPathGenerator
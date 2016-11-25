if not torch then require 'torch' end
if not class then class = require 'class' end
local ImgFinder = require 'DataProvider/ImgFinder'

local ImgFinderTwoPath = class("ImgFinderTwoPath", "ImgFinder")
function ImgFinderTwoPath:__init()
  ImgFinder.__init(self)
end
function ImgFinderTwoPath:launch(train_path, val_path, classToIdx)
  local valImagePath, valImageClass = self:optimize(self:findImages(val_path, classToIdx))
  local trainImagePath, trainImageClass = self:optimize(self:findImages(train_path, classToIdx))
  return valImagePath, valImageClass, trainImagePath, trainImageClass
end
return ImgFinderTwoPath
if not torch then require 'torch' end
if not class then class = require 'class' end
local ImgFinder = require 'DataProvider/ImgFinder'

local ImgFinderOnePath = class("ImgFinderOnePath", "ImgFinder")
function ImgFinderOnePath:__init()
  ImgFinder.__init(self)
end
function ImgFinderOnePath:launch(train_path, classToIdx)
  return self:findImages(train_path, classToIdx)
end
return ImgFinderOnePath
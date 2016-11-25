--[[ImageProvider provides a transformed images based on received index]]--
if not torch then require 'torch' end
if not class then class = require 'class' end
local ImageGenerator = require 'ImageGenerator'

local ImageProvder = class('ImageProvider')
function ImageProvder:__init(opt, img_paths)
  self.ImageGenerator = ImageGenerator(opt)
  self.ImagePaths = img_paths
  self.name = 'image_data'
end
function ImageProvder:exec(atIndex)
  return self.ImageGenerator:exec(self.ImagePaths[atIndex])
end
return ImageProvder

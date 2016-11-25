if not torch then require 'torch' end
if not image then require 'image' end
--if not class then class = require 'class' end
if not paths then require 'paths' end
local M = {}
local ImageLoader = torch.class('ImageLoader', M)
function ImageLoader:__init(set_path)
  assert(paths.dirp(set_path),"Error: Provided path (" .. set_path .. ") doesn't exit.")
  self.set_path = set_path
end
function ImageLoader:exec(file)
  local path = paths.concat(self.set_path, file)
  if not paths.filep(path) then
    --print("Warning: Requested file dosen't exist. Requested file is: " .. path)
    return nil
  end
  return image.load(path):float()
end
return M.ImageLoader
--[[
  local imgProvider = ImageLoader('/Users/aponamaryov/Downloads/training/Aaron_Ashmore')
  local img = imgProvider:exec('25_Aaron_Ashmore_0003.jpg')
  print("Loaded image: ", #img)
  image.display({img})
  print("test is finished successfully")]]--

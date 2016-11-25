if not torch then require 'torch' end
if not class then class = require 'class' end
local sys = require 'sys'
local ffi = require 'ffi'
local image = require 'image'

local ImgFinder = class("ImgFinder")
function ImgFinder:__int()

end
function ImgFinder:findImages(dir, classToIdx)
  local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
  -- Options for the GNU and BSD find command
  for i=2,#extensionList do
    findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
  end
  ----------------------------------------------------------------------
  local imagePath = torch.CharTensor()
  local imageClass = torch.LongTensor()
  -- Find all the images using the find command
  local f = io.popen('find -L ' .. dir .. findOptions)

  local maxLength = -1
  local imagePaths = {}
  local imageClasses = {}

  -- Generate a list of all the images and their class
  while true do
    local line = f:read('*line')
    if not line then break end

    local className = paths.basename(paths.dirname(line))
    --local filename = paths.basename(line)
    --local path = className .. '/' .. filename
    local path = line

    local classId = classToIdx[className]
    assert(classId, 'class not found: ' .. className)

    table.insert(imagePaths, path)
    table.insert(imageClasses, classId)

    maxLength = math.max(maxLength, #path + 1)
  end

  f:close()


  return imagePaths, imageClasses, maxLength
end
function ImgFinder:optimize(imagePaths, imageClasses, maxLength)
  -- Convert the generated list to a tensor for faster loading
  local nImages = #imagePaths
  local imagePath = torch.CharTensor(nImages, maxLength):zero()
  for i, path in ipairs(imagePaths) do
    ffi.copy(imagePath[i]:data(), path)
  end

  local imageClass = torch.LongTensor(imageClasses)
  return imagePath, imageClass
end
return ImgFinder
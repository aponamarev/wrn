if not torch then require 'torch' end
--if not class then class = require 'class' end
if not image then require 'image' end
local M = {}
local Scale = torch.class('Scale', M)
function Scale:__init(scale)
  assert(scale, "Error: scaling factor wasn't provided for Scale class initialization. Provide a number (n) or a table {H,W}!")
  self.multiple = 1
  if type(scale) == 'number' then
    self.scale = {H = scale, W = scale}
  else
    assert(scale.H and scale.W, "Error: scaling factor wasn't provided for Scale class initialization. Provide a number (n) or a table {H,W}!")
    self.scale = scale
  end
end
function Scale:exec(img)
  return self.__f(img, self.scale.H, self.scale.W)
end
function Scale.__f(img, H, W)
  local d = img:dim()
  local s = img:size()
  local t = img:type()
  if d < 4 then
    s = torch.LongStorage({1, s[1], s[2], s[3]})
    img = torch.Tensor(s):type(t):copy(img)
  end
  local img_table = torch.Tensor(s[1], s[2], H, W):type(t)
  for i = 1, s[1] do
    img_table[i]:copy(image.scale(img[i], W, H))
  end
  return img_table
end
return M.Scale

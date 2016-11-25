if not torch then require 'torch' end
if not paths then require 'paths' end
local M = {}
local Whitening = torch.class('Whitening', M)
function Whitening:__init(path)
end
function Whitening:exec(img)
  if not img then
    return nil
  end
  local c = img:size(1)
  local new_img = img:view(c, -1)
  local mean = torch.mean(new_img, 2):squeeze()
  local std = torch.std(new_img, 2):squeeze()
  for i=1,3 do
    img[i]:add(-mean[i])
    img[i]:div(std[i] )
  end
  return img
end
return M.Whitening
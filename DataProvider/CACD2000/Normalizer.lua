if not torch then require 'torch' end
if not paths then require 'paths' end
local M = {}
local Normalizer = torch.class('Normalizer', M)
function Normalizer:__init(path)
  assert(paths.filep(path), "Error: incorrect path was provided to Normalizer. File " .. path .. " doesn't exist")
  self.norm = torch.load(path, 'ascii')
end
function Normalizer:exec(img)
  if not img then return nil end
  for i=1,3 do
    img[i]:add(-self.norm.mean[i])
    img[i]:div(self.norm.std[i])
  end
  return img
end
return M.Normalizer
if not torch then require 'torch' end
--if not class then class = require 'class' end
local M = {}
local RandomIndex = torch.class('RandomIndex', M)
function RandomIndex:__init(size)
  self.size = size
  self.index = {}
  self.index_size = 0
end
function RandomIndex:get()
  return math.random(self.size)
end
return M.RandomIndex
--[[local r = RandomIndex(5)
for i = 1, 20 do
  print(r:get())
end
print("Done")]]--


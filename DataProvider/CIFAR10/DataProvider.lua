if not torch then require 'torch' end
if not class then class = require 'class' end
local MiniBatch = require 'DataProvider/CIFAR10/MiniBatch'
local MultiThreading = require 'DataProvider/CACD2000/MultiThreading'

local DataProvider = class('DataProvider')
function DataProvider:__init(descriptor, opt)
  assert(opt, "Error: Opt parameter wasn't providedto DataProvider at initialization")
  self.m_flag = false
  self.minibatch = MiniBatch(descriptor, opt)
  if self.m_flag then
    self.multithreading = MultiThreading(function()
        return {__threadid, thread_mini:get()}
      end, opt.n_batches, opt.n_batches,
      function()
        if not torch then require 'torch' end
        MiniBatch = require 'DataProvider/CACD2000/MiniBatch'
      end,
      function()
        math.randomseed(__threadid)
        thread_mini = self.minibatch
      end)
  end
end
function DataProvider:get()
  local result
  if self.m_flag then
    result = self.multithreading:get()
    collectgarbage()
  else
    result = self.minibatch:get()
    collectgarbage()
  end
  return result
end
return DataProvider--[[
local opt = {
  gen = 'Results',
  dataset_path = '/Users/aponamaryov/Downloads/CACD2000',
  mean_std_path = 'Results/norm.t7',
  train_descriptor = 'Results/cacd.t7', 
  val_descriptor = 'Results/small.t7', 
  scale = 128,
  --flip = {vertircal = nil},
  --crop = 128,
  --fivecrop = 128,
  batch_size = 20,
  n_batches = 8
}
local d = DataProvider(opt.train_descriptor, opt)
local function calc_mean(img)
  local c = img:size(1)
  img = img:view(c, -1)
  local mean = torch.mean(img, 2):float()
  return mean
end
local global_mean = torch.Tensor(3):zero():float()
local counter = 0
for i = 1, 10000 do
  local result = d:get()
  local size = result.data:size(1)
  for j = 1, size do
    counter = counter + 1
    local mean = calc_mean(result.data[j])
    global_mean:add(mean)
    if counter > 1 and counter % 1000 == 0 then
      print("Iteration " .. (( i - 1 )*j + j) .. " global mean: ")
      print(global_mean)
    end
  end
end--]]--
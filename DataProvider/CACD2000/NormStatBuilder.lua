--[[class is responsible for creating a normalization stats for the training set. The class is initialized with mean norm tables to accumulate stats there.]]--

if not torch then require 'torch' end
if not class then class = require 'class' end
local NormalizeClass = class('NormalizeClass')
function NormalizeClass:__init()
  self.n = 0
  self.mean = torch.zeros(3)
  self.std_sq = torch.zeros(3)
end
function NormalizeClass:calc_normstats(img)
  self.n = self.n + 1
  local c = img:size(1)
  img = img:view(c, -1)
  local mean = torch.mean(img, 2):squeeze()
  self.mean:mul((self.n-1)/self.n)
  self.mean:add(mean/self.n)

  for i = 1, c do
    img[i]:add(-mean[i])
  end
  img:pow(2)
  img = torch.mean(img, 2)
  self.std_sq:mul((self.n-1)/self.n)
  self.std_sq:add(img/self.n)
end
function NormalizeClass:export()
  return self.mean:float(), self.std_sq:sqrt():float()
end

local N = NormalizeClass()
local DataLoader = require('DataProvider/CACD2000/DataProvider')
local dataLoader_opt = {
  gen = 'Results',
  dataset_path = '/Users/aponamaryov/Downloads/CACD2000',
  train_descriptor = '/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/cacd.t7', 
  val_descriptor = '/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/val_cacd.t7', 
  scale = 250,
  batch_size = 100,
  n_batches = 5
}
local D = DataLoader(dataLoader_opt.train_descriptor,dataLoader_opt)
local n = 0
local minibatch
for i = 1, 10000 do
  if n == 0 then
    minibatch = D:get()
    n = minibatch.data:size(1)
  end
  N:calc_normstats(minibatch.data[n])

  if i % 2500 == 0 then
    print(i .. " Images processed for mean calculation. Global mean is:")
    print(N.mean)
    print("Global std is:")
    print(torch.sqrt(N.std_sq))
  end
  n = n - 1
end
local norm_stat = {mean = nil, std = nil}
norm_stat.mean, norm_stat.std = N:export()
torch.save('Results/norm.t7', norm_stat, 'ascii')

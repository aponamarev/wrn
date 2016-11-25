if not torch then require 'torch' end
if not paths then require 'paths' end

local function r_line(f)
  local data = {}
  local check =  true

  local l = f:readString("*l")
  for token in string.gmatch(l, "%d+") do
    data[#data+1] = token .. '.png'
  end
  for token in string.gmatch(l, "%a+") do
    data[#data+1] = token
  end
  return data, check
end

local file = torch.DiskFile('/Users/aponamaryov/Downloads/trainLabels.csv','r')

local data = r_line(file)
local n = 0
local resutl = {file_name = {}, identity_id = {}}
local check = true
for n = 1, 49999 do
  data, check = r_line(file)
  resutl.file_name[n] = data[1]
  resutl.identity_id[n] = data[2]
  if n % 10000 == 0 then print(n .. ' items were transformed.') end
end

torch.save('Results/cifar_descriptor.t7', resutl, 'ascii')
print("Cifar export is done!")
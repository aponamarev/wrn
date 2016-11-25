require 'cltorch'
print('num devices: ', cltorch.getDeviceCount())
local props = cltorch.getDeviceProperties(1)
for k, v in pairs(props) do
  print(k, v)
end
cltorch.setDevice(1)
print('current device: ', cltorch.getDevice())
cltorch.synchronize()
cltorch.finish() -- alias for synchronize()
print('create c=CLStorage()')
local c = torch.ClStorage{4,9,2}
print('fill c tensor with 7s')
c:fill(7)
print('c content:')
print(c)
print(torch.type(c))


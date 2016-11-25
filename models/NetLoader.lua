if not torch then require 'torch' end
if not class then class = require 'class' end
if not paths then require 'paths' end

local NetLoader = class('NetLoader')
function NetLoader:__init(opt)
  self.net = nil
  self.opt = nil
  local mode = opt.pretrainedNet_mode or 'ascii'
  if opt.pretrainedNet_path and paths.filep(opt.pretrainedNet_path) then
    print("Loading Net: " .. opt.pretrainedNet_path)
    self.net = torch.load(opt.pretrainedNet_path, mode)
  end
  if opt.pretrainedOpt_path and paths.filep(opt.pretrainedOpt_path) then
    print("Loading Options: " .. opt.pretrainedOpt_path)
    self.opt = torch.load(opt.pretrainedOpt_path, mode)
  end
end
function NetLoader:launch()
  if not self.net then print("No pretrained net was provided. A new net will be generated") end
  return self.net, self.opt
end
return NetLoader
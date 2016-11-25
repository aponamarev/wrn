--[[
NetSave - saves the net and prints out save-status
Required Objects:
Mediator object needed for initialization of the NetSave

Related Objects
Parrent: AbstractObj (part of mediator.lua module)

Required Functions:
:__init(mediator) init super with mediator and path = self.netLoader_opt.path, file_name = self.netLoader_opt.file_name
:send(message) method inherited from AbstrackObj class
:recieve(msg) saves the network provided in the msg
]]--
if not torch then require 'torch' end
if not paths then require 'paths' end
if not class then class = require 'class' end
if not Mediator or not AbstractObject then require 'trainer/mediator' end --provides class AbstractObject and Mediator

local NetSave = class('NetSave', 'AbstractObject')
function NetSave:__init(path, file_name, mediator, mode)
  AbstractObject.__init(self, mediator)
  self.net_fullPath = paths.concat(path, file_name .. '.t7')
  self.opt_fullPath = paths.concat(path, file_name .. '_opt.t7')
  if not mode then
    self.mode = 'binary'
  else
    self.mode = 'ascii'
  end
end
function NetSave:receive(msg)
  --msg should contain .key and .value
  --saves the network provided in the msg.value
  if msg.key == "save" then
    print("Staring the process to save the net.")
    torch.save(self.net_fullPath, msg.value.net, self.mode)
    torch.save(self.opt_fullPath, msg.value.opt, self.mode)
    collectgarbage()
    print("Net was saved to " .. self.net_fullPath)
  end
end

return NetSave
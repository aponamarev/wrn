if not torch then require 'torch' end

local M = {}
local Protocol = torch.class("Protocol", M)
function Protocol:__init()
  self.name = "Protocol to be filled"
end
function Protocol:updateName(name)
  assert(false, "Protocol:updateName() should be filled in child class")
end
local protocol = M.Protocol()
local Child = torch.class("Child", protocol)
function Child:__init()
  Protocol.__init(self)
end
function Child:updateName(name)
  self.name = name
end
local child = protocol.Child()
child:updateName("bubbles")
print(child.name)

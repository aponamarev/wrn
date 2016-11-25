require 'torch'
require 'paths'
matio = require 'matio'
ffi = require 'ffi'
class = require 'class'
require 'sys'

local source_file = 'kzg]/home/ubuntu/Downloads/celebrity2000_meta.mat'
local original_path = '/home/ubuntu/CACD2000'
local store_directory = '/home/ubuntu/Model/wrn/Results'

local Mover = class("Mover")
function Mover:__init(source, pictures_path, destination)
  assert(paths.filep(source),"Error: proiveded source file doesn't exist: " .. source)
  print("Starting to load matlab file.")
  local meta = matio.load(source)
  print("File was loaded successfully.")
  self.pictures_path = pictures_path
  self.destination = destination
  self.name = meta.celebrityData.name
  self.identity_id = meta.celebrityData.identity
  self.file_id = meta.celebrityImageData.identity
  self.birth_year = meta.celebrityData.birth
  self.lfw = meta.celebrityData.lfw
  self.age = meta.celebrityImageData.age
  self.picture_year = meta.celebrityImageData.year
  self.file_name = meta.celebrityImageData.name
  self.__length = self.file_id:size(1)
  print(self.__length .. " files will be moved")
end
function Mover:__move(index)
  local t = self.file_name[index]
  local file_name = ffi.string(t:data(), t:size(2))
  local id = self.file_id[index][1]
  local age = self.age[index][1]
  local picture_year = self.picture_year[index][1]
  local birth_year = self.birth_year[id][1]
  local name = ffi.string(self.name[id]:data())
  local destination_path = paths.concat(self.destination, name)
  local destination_file = paths.concat(destination_path, file_name)
  if not paths.dirp(destination_path) then
    paths.mkdir(destination_path)
  end
  local original_file = paths.concat(self.pictures_path, file_name)
  local not_found = nil
  if paths.filep(original_file) then
    sys.execute("mv " .. original_file .. " " .. destination_file)
  else
    not_found = original_file
    print("The following file wasn't found: " .. original_file)
  end
  return destination_file, name, age, picture_year, birth_year, not_found
end
function Mover:exec()
  print("Starting the file transfer process!")
  local M = {}
  local description = {}
  local not_found = {}
  for i = 1, self.__length do
    if i % 500 == 0 then
      print(#description .. " files were moved, and " .. #not_found .. " files were missed.")
    end
    local __error = nil
    M.file, M.name, M.age, M.picture_year, M.birth_year, __error = self:__move(i)
    if __error then
      not_found[#not_found+1] = M
    else
      description[#description+1] = M
    end
  end
  print("File transfer is completed with total of " .. #description .. " files transfered and " .. #not_found .. " files missed. Saving information.")
  return description, not_found
end
function Mover:transform()
  local t_f = function(index)
    local t = self.file_name[index]
    local file_name = ffi.string(t:data(), t:size(2))
    local id = self.file_id[index][1]
    local age = self.age[index][1]
    --local picture_year = self.picture_year[index][1]
    --local birth_year = self.birth_year[id][1]
    local name = ffi.string(self.name[id]:data())
    return file_name, age, id--, picture_year, birth_year
  end
  local classes = function(index)
    local class_name = ffi.string(self.name[index]:data())
    return class_name
  end

  local resulting_data = {file_name = {}, age = {}, identity_id = {}, classes = {}}--picture_year = {}, birth_year = {}, 
  print("start transformation")
  for i = 1, self.__length do
    resulting_data.file_name[i], resulting_data.age[i], resulting_data.identity_id[i] = t_f(i)--resulting_data.picture_year[i], resulting_data.birth_year[i], 
    if i % 25000 == 0 then print(i .. " items out of " .. self.__length .. " were transformed") end
  end
  resulting_data.size = self.__length
  resulting_data.num_classes = self.identity_id:size(1)
  print("Transforing classes info.")
  for i = 1, resulting_data.num_classes do
    resulting_data.classes[i] = classes(i)
  end
  print("Transformation is finished. Returning the data")
  return resulting_data
end
function Mover:val(data)
  local size = data.size
  local val_data = {file_name = {}, age = {}, identity_id = {}, classes = {}}
  for i = 1, math.ceil(data.size*0.05) do
    local ri = math.random(1, size)
    size = size - 1
    val_data.file_name[i] = table.remove(data.file_name, ri)
    val_data.age[i] = table.remove(data.age, ri)
    val_data.identity_id[i] = table.remove(data.identity_id, ri)
    if i % 2500 == 0 then print("Validation data: " .. i .. " transactions done. The size of val_data.file_name is ", #val_data.file_name) end
  end
  data.size = size
  for k, v in pairs(data.classes) do
    val_data.classes[k] = v
    if k % 500 == 0 then print("Classes transfered: " .. k) end
  end
  val_data.size = #val_data.file_name
  val_data.num_classes = #val_data.classes
  print("Total number of classes: " .. val_data.num_classes)
  return val_data, data
end
local move = Mover(source_file, original_path, store_directory)
local data = move:transform()
local val_data
val_data, data = move:val(data)
torch.save("/home/ubuntu/Model/wrn/Results/cacd.t7", data, 'ascii')
torch.save("/home/ubuntu/Model/wrn/Results/val_cacd.t7", val_data, 'ascii')

--[[
Create a dictinary with: 
file_name = {},
age = {},
classes	= {},
size = {},
identity_id = {},
num_classes = {}

fill in the content of the disctinary by parsing files
]]--

require 'paths'
if not(torch) then require 'torch' end

local path_to_cacd2000_folder = '/Users/aponamaryov/Downloads/CACD2000'
local save_descriptor_to = '/Users/aponamaryov/Google_Drive/AI_Project/Torch_Code/ImgClassificationPipeline-Torch/Results/cacd.t7'
local all_files = paths.files(path_to_cacd2000_folder)

local cacd = {
  file_name = {},
  age = {},
  classes	= {},
  size = 0,
  identity_id = {},
  num_classes = 0
}

function match_element(t, elem)
  local r = {}
  for i, c in pairs(t) do
    if c == elem then
      table.insert(r, #r+1, i)
    end
  end
  if #r==0 then r=nil end
  return r
end
function extract_age_and_name(s)
  --local age, name = extract_age_and_name(file_name)
  --extract tonumber(age) and name from the file name
  local underscores = {}
  local match_ = s:find("_")
  while not(match_==nil) do
    table.insert(underscores, #underscores+1, match_)
    match_ = s:find("_", match_ + 1)
  end
  --find age
  local age = tonumber(s:sub(1,underscores[1]-1))
  --assert(#underscores>2,"Error: could not find underscore symbols for " .. s .. " file number: " .. cacd.size+1)
  local name = ""
  if #underscores>2 then
    name = s:sub(underscores[1]+1,underscores[3]-1)
  elseif #underscores==2 then
    name = s:sub(underscores[1]+1,underscores[2]-1)
  else
    assert(false,"Error: could not find underscore symbols for " .. s .. " file number: " .. cacd.size+1)    
  end
  return age, name
end

function parser(file_name, descriptor)
  local age, name = extract_age_and_name(file_name)
  local class_id = match_element(descriptor.classes, name)
  if class_id then
    class_id = table.unpack(class_id)
  else
    --add another name to the table and modify class_id
    class_id = #descriptor.classes + 1
    table.insert(descriptor.classes, class_id, name)
    descriptor.num_classes = class_id
  end
  local fileid = #descriptor.file_name+1
  table.insert(descriptor.file_name, fileid, file_name)
  table.insert(descriptor.age, fileid, age)
  table.insert(descriptor.identity_id, fileid, class_id)
  descriptor.size = descriptor.size + 1
end
--run the script
for file_name in all_files do
  if file_name:sub(-3,-1) == "jpg" then
    parser(file_name, cacd)
  else
    print("wrong file name:", file_name)
  end
  if (cacd.size>2) and (cacd.size % 1000 == 0) then
    print(cacd.size, "files were processed.", cacd.num_classes, " classes of classes were found")
  end
end
print("all " .. cacd.size .. " files were processed! Start saving.")
torch.save(save_descriptor_to, cacd)
print("cacd.t7 file was saved successfully!")

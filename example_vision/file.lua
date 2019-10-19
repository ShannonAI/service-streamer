function read_bin_file(path)
    local file, errorMessage = io.open(path, "rb")
    if not file then 
        error("Could not read the file:" .. errorMessage .. "\n")
    end

    local content = file:read "*all"
    file:close()
    return content
end

local Boundary = "----WebKitFormBoundaryePkpFF7tjBAqx29L"
local BodyBoundary = "--" .. Boundary
local LastBoundary = "--" .. Boundary .. "--"

local CRLF = "\r\n"

local Filename = "cat.jpg"
local FileBody = read_bin_file(Filename)

local ContentDisposition = "Content-Disposition: form-data; name=\"file\"; filename=\"" .. Filename .. "\""

wrk.method = "POST"
wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. Boundary
wrk.body = BodyBoundary .. CRLF .. ContentDisposition .. CRLF .. CRLF .. FileBody .. CRLF .. LastBoundary

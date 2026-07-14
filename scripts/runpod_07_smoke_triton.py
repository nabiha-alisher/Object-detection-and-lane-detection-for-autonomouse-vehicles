import tritonclient.http as httpclient
c=httpclient.InferenceServerClient('localhost:8000')
print('live:', c.is_server_live())
print('ready:', c.is_server_ready())
for m in ['yolopx','depth_metric','traffic']:
    print(m, 'ready:', c.is_model_ready(m))
    print(c.get_model_metadata(m))

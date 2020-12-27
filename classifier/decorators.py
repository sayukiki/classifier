"""デコレーターをasync/awaitに対応させた。"""

from functools import wraps
from django.http import HttpResponseNotAllowed
from django.utils.log import log_response

def require_http_methods(request_method_list):
    def decorator(view_func):
        async def wrapped_view(request, *args, **kwargs):
            if request.method not in request_method_list:
                response = HttpResponseNotAllowed(request_method_list)
                log_response(
                    'Method Not Allowed (%s): %s', request.method, request.path,
                    response=response,
                    request=request,
                )
                return response
            return await view_func(request, *args, **kwargs)
        return wraps(view_func)(wrapped_view)
    return decorator
"""Retain JSON decimal literals before provider SDKs normalize their objects."""
import json
from decimal import Decimal


def exact_event(event):
    event.json = lambda: json.loads(event.data, parse_float=Decimal)
    return event


async def exact_response_json(response):
    if getattr(response, '_power_loop_exact_json', False):
        return
    original = response.json

    def parse(**options):
        options.setdefault('parse_float', Decimal)
        return original(**options)

    response.json = parse
    response._power_loop_exact_json = True


def configure_http_client(client):
    hooks = client.event_hooks.setdefault('response', [])
    if exact_response_json not in hooks:
        hooks.append(exact_response_json)
    return client

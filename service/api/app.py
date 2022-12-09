import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import uvloop
from fastapi import FastAPI

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from ..userknn import (
    OfflineRecommender,
    OnlineUserKnnRecommender,
    PopularItemsPadder,
)
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)

    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs
    app.state.models = {
        'offline_tfidf_idf_10': OfflineRecommender(
            config.offline_tfidf_idf_10,
            Path(config.models_dir)),
        'online_tfidf_idf_10': OnlineUserKnnRecommender(
            config.online_tfidf_idf_10,
            Path(config.models_dir),
            config.train_name),
        'offline_mf_ann': OfflineRecommender(
            config.offline_mf_ann,
            Path(config.models_dir)),
    }
    postprocessing: Optional[PopularItemsPadder] = None
    if config.postprocessing == 'PopularPadding':
        postprocessing = PopularItemsPadder(
            k=config.k_recs,
            popular_name=config.popular_name,
            models_dir=Path(config.models_dir)
            )
    app.state.postprocessing = postprocessing

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app

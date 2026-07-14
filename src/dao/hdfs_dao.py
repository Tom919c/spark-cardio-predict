"""HDFS 访问抽象：本地模式下不会在导入时依赖 pyspark 或 hdfs 包。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit


class HDFSClient(Protocol):
    def makedirs(self, path: str) -> None: ...
    def mkdirs(self, path: str) -> None: ...
    def upload(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> str: ...
    def exists(self, path: str) -> bool: ...
    def status(self, path: str, strict: bool = True): ...
    def write(self, path: str, data, overwrite: bool = True): ...
    def read(self, path: str): ...
    def delete(self, path: str, recursive: bool = False) -> bool: ...


class HDFSDao:
    def __init__(self, client: HDFSClient | None = None):
        self.client = client

    def is_available(self) -> bool:
        return self.client is not None

    def ensure_dir(self, path: str) -> None:
        if self.client:
            # hdfs.InsecureClient uses makedirs; mkdirs is retained for simple
            # compatible clients used by integrations and tests.
            if hasattr(self.client, "makedirs"):
                self.client.makedirs(path)
            elif hasattr(self.client, "mkdirs"):
                self.client.mkdirs(path)
            else:
                raise RuntimeError("HDFS client does not support directory creation.")

    def upload_file(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> str:
        if not self.client:
            raise RuntimeError("HDFS client is not configured.")
        return self.client.upload(local_path, hdfs_path, overwrite=overwrite)

    def exists(self, path: str) -> bool:
        if not self.client:
            return False
        # hdfs.InsecureClient exposes status(..., strict=False), not exists().
        if hasattr(self.client, "exists"):
            return bool(self.client.exists(path))
        return self.client.status(path, strict=False) is not None

    def status(self, path: str, strict: bool = True):
        if not self.client:
            return None
        return self.client.status(path, strict=strict)

    def write_bytes(self, path: str, data: bytes, overwrite: bool = True) -> None:
        if not self.client:
            raise RuntimeError("HDFS client is not configured.")
        with self.client.write(path, overwrite=overwrite) as writer:
            writer.write(data)

    def merge_files(self, chunk_paths: list[str], target_path: str) -> str:
        """流式合并 HDFS 分片，避免把完整文件拉回 Flask。"""
        if not self.client:
            raise RuntimeError("HDFS client is not configured.")
        with self.client.write(target_path, overwrite=True) as writer:
            for chunk_path in chunk_paths:
                with self.client.read(chunk_path) as reader:
                    while True:
                        block = reader.read(1024 * 1024)
                        if not block:
                            break
                        writer.write(block)
        return target_path

    def delete(self, path: str, recursive: bool = False) -> bool:
        if not self.client:
            return False
        return bool(self.client.delete(path, recursive=recursive))

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if not self.client:
            raise RuntimeError("HDFS client is not configured.")
        with self.client.read(path) as reader:
            content = reader.read()
        return content.decode(encoding) if isinstance(content, bytes) else content

    @classmethod
    def from_config(cls, config):
        if str(config.get("DATA_MODE", "local")).lower() != "hdfs":
            return cls()
        try:
            from hdfs import InsecureClient

            web_url = config.get("HDFS_WEB_URL", "http://localhost:9870")
            hosts = [urlsplit(item).hostname for item in str(web_url).split(";")]
            datanode_host = config.get("HDFS_DATANODE_HOST", "")
            if datanode_host:
                class ConfiguredInsecureClient(InsecureClient):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._namenode_hosts = {host.lower() for host in hosts if host}
                        self._datanode_host = str(datanode_host).strip()
                        import requests

                        parent_session = self._session
                        datanode = self._datanode_host
                        namenodes = self._namenode_hosts

                        class RedirectingSession(requests.Session):
                            @staticmethod
                            def _rewrite(url):
                                parsed = urlsplit(url)
                                if not parsed.hostname or parsed.hostname.lower() in namenodes:
                                    return url
                                return urlunsplit(
                                    (
                                        parsed.scheme,
                                        f"{datanode}:{parsed.port}" if parsed.port else datanode,
                                        parsed.path,
                                        parsed.query,
                                        parsed.fragment,
                                    )
                                )

                            def request(self, method, url, **kwargs):
                                # hdfs._Request disables redirects for CREATE so it can
                                # stream the body itself; OPEN needs one explicit follow.
                                allow_redirects = kwargs.pop("allow_redirects", True)
                                kwargs["allow_redirects"] = False
                                response = super().request(
                                    method, self._rewrite(url), **kwargs
                                )
                                location = response.headers.get("location")
                                if allow_redirects and location and response.is_redirect:
                                    response.close()
                                    response = super().request(
                                        method,
                                        self._rewrite(location),
                                        **kwargs,
                                    )
                                return response

                        self._session = RedirectingSession()
                        self._session.headers.update(parent_session.headers)
                        self._session.params.update(parent_session.params)

                    def _request(self, method, url, **kwargs):
                        parsed = urlsplit(url)
                        if parsed.hostname and parsed.hostname.lower() not in self._namenode_hosts:
                            url = urlunsplit(
                                (
                                    parsed.scheme,
                                    f"{self._datanode_host}:{parsed.port}" if parsed.port else self._datanode_host,
                                    parsed.path,
                                    parsed.query,
                                    parsed.fragment,
                                )
                            )
                        return super()._request(method, url, **kwargs)

                client = ConfiguredInsecureClient(web_url, user=config.get("HDFS_USER"))
            else:
                client = InsecureClient(web_url, user=config.get("HDFS_USER"))
            return cls(client)
        except (ImportError, OSError, ValueError):
            return cls()

    @staticmethod
    def local_safe_name(filename: str) -> str:
        return Path(filename).name

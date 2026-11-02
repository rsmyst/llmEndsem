from __future__ import annotations

from src.matp.resources import ResourceResolver, normalize_path


def test_windows_path_normalisation():
    path = normalize_path(r"C:\Users\Test\Project\Config.YAML")
    assert "\\" not in path
    assert path.endswith("config.yaml")


def test_sql_read_and_write():
    r = ResourceResolver()
    read = r.extract("sql_query", {"query": "SELECT * FROM users", "table": "users"}, "SUCCESS", "[]")
    assert [(a.resource_id, a.mode) for a in read] == [("db:users", "read")]

    write = r.extract("sql_query", {"query": "ALTER TABLE users ADD COLUMN x TEXT"}, "SUCCESS", "ok")
    assert (write[0].resource_id, write[0].mode) == ("db:users", "write")

    delete = r.extract("sql_query", {"query": "DROP TABLE users"}, "SUCCESS", "ok")
    assert delete[0].mode == "delete"


def test_http_and_files():
    r = ResourceResolver()
    get = r.extract("http_request", {"endpoint": "https://api.example.com/v1/users", "method": "GET"}, "SUCCESS", "{}")
    assert get[0].resource_id == "api:api.example.com/v1/users"
    assert get[0].mode == "read"

    post = r.extract("http_request", {"endpoint": "https://api.example.com/v1/users", "method": "POST"}, "SUCCESS", "{}")
    assert post[0].mode == "write"

    write = r.extract("write_file", {"path": "src/app.py"}, "SUCCESS", "print(1)")
    assert write[0].mode == "write"
    assert write[0].resource_id.startswith("file:")


def test_shell_accesses():
    r = ResourceResolver(workspace="/repo")
    accesses = r.extract("bash_exec", {"command": "cat a.txt > b.txt && rm c.txt"}, "SUCCESS", "ok")
    modes = {(a.resource_id.split("/")[-1], a.mode) for a in accesses}
    assert ("a.txt", "read") in modes
    assert ("b.txt", "write") in modes
    assert ("c.txt", "delete") in modes


def test_git_accesses():
    r = ResourceResolver(workspace="/repo")
    read = r.extract("git_command", {"command": "git status"}, "SUCCESS", "ok")
    write = r.extract("git_command", {"command": "git commit -am x"}, "SUCCESS", "ok")
    assert read[0].mode == "read"
    assert write[0].mode == "write"

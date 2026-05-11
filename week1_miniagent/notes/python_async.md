# Python 异步编程

## async/await 基础

Python 3.5+ 引入 async/await 语法。

```python
import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)
    return f"data from {url}"

async def main():
    result = await fetch_data("http://example.com")
    print(result)

asyncio.run(main())
```

## 事件循环

事件循环是 asyncio 的核心，负责调度和执行异步任务。

- `asyncio.get_event_loop()`: 获取当前事件循环
- `asyncio.run()`: Python 3.7+ 推荐的启动方式

## 并发执行

### gather vs wait

`asyncio.gather()` 并发运行多个协程，返回结果列表。
`asyncio.wait()` 更灵活，可以设置超时和返回条件。

```python
results = await asyncio.gather(
    fetch_data("url1"),
    fetch_data("url2"),
    fetch_data("url3")
)
```

## 线程 vs 协程

| 特性 | 线程 | 协程 |
|------|------|------|
| 切换开销 | 大（内核态） | 小（用户态） |
| 内存占用 | ~8MB/线程 | ~KB |
| 适合场景 | CPU密集 | IO密集 |

# simple tester to load the echo service with concurrent requests

import asyncio
import aiohttp
import time
import json
import argparse
import statistics

URL = "http://localhost:30080/echo"

async def send_request(session, req_id):
    payload = {"test": "data", "id": req_id}
    start_time = time.time()
    try:
        async with session.post(URL, json=payload) as response:
            await response.read()
            status = response.status
    except Exception as e:
        status = str(e)
    
    end_time = time.time()
    latency = end_time - start_time
    return {
        "request_id": req_id,
        "start_time": start_time,
        "latency": latency,
        "status": status
    }

async def run_load_test(total_requests, concurrency):
    print(f"Starting load test: {total_requests} requests with concurrency {concurrency}")
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        sem = asyncio.Semaphore(concurrency)
        
        async def bound_send(req_id):
            async with sem:
                return await send_request(session, req_id)

        tasks = [bound_send(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
    return results

def save_results(data, filename="latency_results.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filename}")

def print_stats(results):
    latencies = [r['latency'] for r in results]
    successful = [r for r in results if r['status'] == 200]
    
    stats = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "average_latency": statistics.mean(latencies) if latencies else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0
    }

    print("\nTest stats")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful Requests: {stats['successful_requests']}")
    print(f"Average Latency: {stats['average_latency']:.4f}s")
    print(f"Min Latency: {stats['min_latency']:.4f}s")
    print(f"Max Latency: {stats['max_latency']:.4f}s")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the echo service.")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--output", type=str, default="latency_results.json", help="Output file for results")
    args = parser.parse_args()

    start_time = time.time()
    results = asyncio.run(run_load_test(args.requests, args.concurrency))
    total_duration = time.time() - start_time
    
    print(f"\nTotal Test Duration: {total_duration:.2f}s")
    stats = print_stats(results)
    
    output_data = {
        "config": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "url": URL
        },
        "summary": stats,
        "total_duration": total_duration,
        "raw_results": results
    }
    
    save_results(output_data, args.output)

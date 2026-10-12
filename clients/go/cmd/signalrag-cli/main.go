// Standalone SignalRAG CLI tool written in Go for low-latency operational control.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/sujanv/SignalRAG/clients/go"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "query":
		queryCmd := flag.NewFlagSet("query", flag.ExitOnError)
		baseURL := queryCmd.String("url", "http://localhost:8000", "SignalRAG API URL")
		topK := queryCmd.Int("k", 3, "Top-K results")
		queryCmd.Parse(os.Args[2:])

		args := queryCmd.Args()
		if len(args) == 0 {
			fmt.Println("Error: query text required")
			os.Exit(1)
		}
		queryText := args[0]

		client := signalrag.NewClient(*baseURL, nil)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		resp, err := client.Query(ctx, signalrag.QueryRequest{
			Query: queryText,
			TopK:  *topK,
		})
		if err != nil {
			fmt.Printf("Query error: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Query: %s\nAnswer: %s\nLatency: %.1fms\n", resp.Query, resp.Answer, resp.LatencyMs)
		fmt.Printf("Citations (%d):\n", len(resp.Citations))
		for _, c := range resp.Citations {
			fmt.Printf("  [%d] Doc: %s\n", c.Index, c.DocumentID)
		}

	case "index":
		indexCmd := flag.NewFlagSet("index", flag.ExitOnError)
		baseURL := indexCmd.String("url", "http://localhost:8000", "SignalRAG API URL")
		indexCmd.Parse(os.Args[2:])

		client := signalrag.NewClient(*baseURL, nil)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		resp, err := client.Index(ctx)
		if err != nil {
			fmt.Printf("Index error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Indexed successfully: %d docs, %d total chunks\n", resp.IndexedDocuments, resp.TotalChunks)

	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: signalrag-cli <command> [options] [arguments]")
	fmt.Println("Commands:")
	fmt.Println("  query [-url http://...] [-k 3] <query text>")
	fmt.Println("  index [-url http://...]")
}

// Standalone concurrent batch loader in Go for high-throughput ingestion.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/sujanv/SignalRAG/clients/go"
)

func main() {
	baseURL := flag.String("url", "http://localhost:8000", "SignalRAG API URL")
	workers := flag.Int("workers", 4, "Number of concurrent goroutines")
	count := flag.Int("count", 20, "Number of synthetic documents to ingest")
	flag.Parse()

	client := signalrag.NewClient(*baseURL, nil)
	t0 := time.Now()

	docChan := make(chan int, *count)
	for i := 1; i <= *count; i++ {
		docChan <- i
	}
	close(docChan)

	var wg sync.WaitGroup
	for w := 1; w <= *workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for docNum := range docChan {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				content := fmt.Sprintf("Document %d: Batch ingested high-concurrency technical note.", docNum)
				_, err := client.AddDocument(ctx, signalrag.DocumentCreateRequest{
					DocumentID: fmt.Sprintf("batch_doc_%d", docNum),
					Content:    content,
				})
				cancel()
				if err != nil {
					fmt.Printf("[Worker %d] Failed doc %d: %v\n", workerID, docNum, err)
				}
			}
		}(w)
	}

	wg.Wait()
	elapsed := time.Since(t0)
	fmt.Printf("Batch ingestion completed: %d documents in %v\n", *count, elapsed)
}

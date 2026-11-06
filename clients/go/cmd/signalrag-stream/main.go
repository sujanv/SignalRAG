// Native Go Server-Sent Events (SSE) streaming client subscriber.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

type SSEMessage struct {
	Event     string                 `json:"event"`
	Content   string                 `json:"content,omitempty"`
	LatencyMs float64                `json:"latency_ms,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

func main() {
	url := flag.String("url", "http://localhost:8000/query/stream", "SignalRAG SSE stream endpoint")
	query := flag.String("q", "Explain streaming retrieval", "Question to stream")
	flag.Parse()

	reqBody, _ := json.Marshal(map[string]interface{}{
		"query": *query,
		"top_k": 3,
	})

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(*url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Fatalf("Stream connection failed: %v", err)
	}
	defer resp.Body.Close()

	fmt.Printf("[Connected to SignalRAG Stream: %s]\n", *url)
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			dataStr := strings.TrimPrefix(line, "data: ")
			var msg SSEMessage
			if err := json.Unmarshal([]byte(dataStr), &msg); err == nil {
				if msg.Event == "token" {
					fmt.Print(msg.Content)
				} else if msg.Event == "done" {
					fmt.Printf("\n[Stream Complete - Latency: %.1fms]\n", msg.LatencyMs)
				}
			}
		}
	}
}

// Package signalrag provides an idiomatic Golang client SDK for the SignalRAG service.
package signalrag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// QueryRequest represents payload sent to /query
type QueryRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

// Citation represents an attribution link
type Citation struct {
	Index      int    `json:"index"`
	DocumentID string `json:"document_id"`
	ChunkID    string `json:"chunk_id"`
	Text       string `json:"text"`
}

// QueryResponse represents the answer and citations from /query
type QueryResponse struct {
	Query     string     `json:"query"`
	Answer    string     `json:"answer"`
	Citations []Citation `json:"citations"`
	LatencyMs float64    `json:"latency_ms"`
}

// DocumentCreateRequest represents payload sent to /documents
type DocumentCreateRequest struct {
	DocumentID string                 `json:"document_id,omitempty"`
	Content    string                 `json:"content"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// DocumentResponse represents document ingestion confirmation
type DocumentResponse struct {
	DocumentID     string                 `json:"document_id"`
	CharacterCount int                    `json:"character_count"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// IndexResponse represents index update results
type IndexResponse struct {
	IndexedDocuments int    `json:"indexed_documents"`
	TotalChunks      int    `json:"total_chunks"`
	Status           string `json:"status"`
}

// Client is a SignalRAG API client
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new SignalRAG Go client
func NewClient(baseURL string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: httpClient,
	}
}

// Query sends a search query and returns the answer with citations
func (c *Client) Query(ctx context.Context, req QueryRequest) (*QueryResponse, error) {
	var resp QueryResponse
	if err := c.post(ctx, "/query", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// AddDocument ingests a new raw text document
func (c *Client) AddDocument(ctx context.Context, req DocumentCreateRequest) (*DocumentResponse, error) {
	var resp DocumentResponse
	if err := c.post(ctx, "/documents", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Index triggers indexing of stored documents
func (c *Client) Index(ctx context.Context) (*IndexResponse, error) {
	var resp IndexResponse
	if err := c.post(ctx, "/index", map[string]interface{}{}, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *Client) post(ctx context.Context, path string, body interface{}, out interface{}) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("signalrag: failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("signalrag: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("signalrag: http request failed: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode >= 300 {
		respBytes, _ := io.ReadAll(res.Body)
		return fmt.Errorf("signalrag: error response %d: %s", res.StatusCode, string(respBytes))
	}

	if out != nil {
		if err := json.NewDecoder(res.Body).Decode(out); err != nil {
			return fmt.Errorf("signalrag: failed to decode response: %w", err)
		}
	}
	return nil
}

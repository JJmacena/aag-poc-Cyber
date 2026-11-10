package actions

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/google/uuid"
)

// TokenType represents the type of honeytoken
type TokenType string

const (
	TokenAWSKey       TokenType = "aws_cred"
	TokenSSHKey       TokenType = "ssh_key"
	TokenAPIKey       TokenType = "api_key"
	TokenDBCredential TokenType = "db_cred"
	TokenGitToken     TokenType = "git_token"
	TokenFile         TokenType = "file"
	TokenDNS          TokenType = "dns"
)

// Persona represents the decoy persona type
type Persona string

const (
	PersonaLegacyWin   Persona = "LEGACY_WIN"
	PersonaK8sNode     Persona = "K8S_NODE"
	PersonaCIRunner    Persona = "CI_RUNNER"
	PersonaIoTGateway  Persona = "IOT_GW"
	PersonaDevWS       Persona = "DEV_WS"
	PersonaCloudFunc   Persona = "CLOUD_FN"
)

// DecoyManager interface for deploying honeytokens
type DecoyManager interface {
	DeployFile(ctx context.Context, path, content string) error
	DeployEnvVar(ctx context.Context, key, value string) error
	DeployDNSRecord(ctx context.Context, name, value string) error
	GetDecoyPath(persona Persona, tokenType TokenType) string
}

// TokenStore interface for tracking deployed honeytokens
type TokenStore interface {
	Store(ctx context.Context, token *Honeytoken) error
	Get(ctx context.Context, tokenID string) (*Honeytoken, error)
	MarkAccessed(ctx context.Context, tokenID string) error
	ListBySession(ctx context.Context, sessionID string) ([]*Honeytoken, error)
}

// NotificationService interface for alerts
type NotificationService interface {
	NotifyHoneytokenTouch(ctx context.Context, token *Honeytoken, sessionID string) error
	NotifyDeployment(ctx context.Context, token *Honeytoken) error
}

// Honeytoken represents a deployed honeytoken
type Honeytoken struct {
	ID          string
	Type        TokenType
	Value       string
	Persona     Persona
	SessionID   string
	DeployedAt  time.Time
	AccessedAt  *time.Time
	AccessCount int
	Metadata    map[string]string
}

// HoneytokenInjector handles honeytoken selection, generation, and deployment
type HoneytokenInjector struct {
	decoyManager     DecoyManager
	tokenStore       TokenStore
	notificationSvc  NotificationService
	personaTokenMap  map[Persona][]TokenType
}

// NewHoneytokenInjector creates a new honeytoken injector
func NewHoneytokenInjector(
	decoyManager DecoyManager,
	tokenStore TokenStore,
	notificationSvc NotificationService,
) *HoneytokenInjector {
	return &HoneytokenInjector{
		decoyManager:    decoyManager,
		tokenStore:      tokenStore,
		notificationSvc: notificationSvc,
		personaTokenMap: defaultPersonaTokenMap(),
	}
}

// defaultPersonaTokenMap defines which token types suit each persona
func defaultPersonaTokenMap() map[Persona][]TokenType {
	return map[Persona][]TokenType{
		PersonaLegacyWin:  {TokenAWSKey, TokenSSHKey, TokenAPIKey, TokenFile, TokenDNS},
		PersonaK8sNode:    {TokenAWSKey, TokenAPIKey, TokenDBCredential, TokenFile, TokenDNS},
		PersonaCIRunner:   {TokenGitToken, TokenAPIKey, TokenAWSKey, TokenFile},
		PersonaIoTGateway: {TokenAPIKey, TokenSSHKey, TokenFile, TokenDNS},
		PersonaDevWS:      {TokenGitToken, TokenSSHKey, TokenAPIKey, TokenAWSKey, TokenFile},
		PersonaCloudFunc:  {TokenAWSKey, TokenAPIKey, TokenDBCredential, TokenFile, TokenDNS},
	}
}

// SelectTokenTypes returns appropriate token types for a persona
func (h *HoneytokenInjector) SelectTokenTypes(persona Persona) []TokenType {
	if types, ok := h.personaTokenMap[persona]; ok {
		return types
	}
	return []TokenType{TokenFile, TokenAPIKey}
}

// GenerateAndInject creates and deploys honeytokens for a session
func (h *HoneytokenInjector) GenerateAndInject(
	ctx context.Context,
	sessionID string,
	persona Persona,
	tokenTypes []TokenType,
) ([]*Honeytoken, error) {
	var tokens []*Honeytoken

	for _, tokenType := range tokenTypes {
		token, err := h.generateToken(tokenType, persona, sessionID)
		if err != nil {
			log.Printf("[Honeytoken] Failed to generate %s: %v", tokenType, err)
			continue
		}

		// Deploy based on type
		if err := h.deployToken(ctx, token); err != nil {
			log.Printf("[Honeytoken] Failed to deploy %s: %v", token.ID, err)
			continue
		}

		// Store
		if err := h.tokenStore.Store(ctx, token); err != nil {
			log.Printf("[Honeytoken] Failed to store %s: %v", token.ID, err)
			continue
		}

		// Notify
		if h.notificationSvc != nil {
			h.notificationSvc.NotifyDeployment(ctx, token)
		}

		tokens = append(tokens, token)
		log.Printf("[Honeytoken] Deployed %s (%s) for session %s", token.ID, tokenType, sessionID)
	}

	return tokens, nil
}

// generateToken creates a honeytoken value based on type
func (h *HoneytokenInjector) generateToken(
	tokenType TokenType,
	persona Persona,
	sessionID string,
) (*Honeytoken, error) {
	token := &Honeytoken{
		ID:         uuid.New().String(),
		Type:       tokenType,
		Persona:    persona,
		SessionID:  sessionID,
		DeployedAt: time.Now(),
		Metadata:   make(map[string]string),
	}

	switch tokenType {
	case TokenAWSKey:
		token.Value = generateAWSCredentials()
		token.Metadata["format"] = "access_key:secret_key"
	case TokenSSHKey:
		token.Value = generateSSHKey()
		token.Metadata["format"] = "private_key_pem"
	case TokenAPIKey:
		token.Value = generateAPIKey(persona)
		token.Metadata["format"] = "bearer_token"
	case TokenDBCredential:
		token.Value = generateDBCredential()
		token.Metadata["format"] = "connection_string"
	case TokenGitToken:
		token.Value = generateGitToken()
		token.Metadata["format"] = "pat_token"
	case TokenFile:
		token.Value = generateHoneyFileContent(persona)
		token.Metadata["format"] = "file_content"
	case TokenDNS:
		token.Value = generateDNSToken(sessionID)
		token.Metadata["format"] = "dns_record"
	default:
		return nil, fmt.Errorf("unknown token type: %s", tokenType)
	}

	return token, nil
}

// deployToken deploys the honeytoken to the environment
func (h *HoneytokenInjector) deployToken(ctx context.Context, token *Honeytoken) error {
	switch token.Type {
	case TokenFile:
		path := h.decoyManager.GetDecoyPath(token.Persona, token.Type)
		return h.decoyManager.DeployFile(ctx, path, token.Value)

	case TokenAWSKey, TokenAPIKey, TokenDBCredential, TokenGitToken:
		// Deploy as environment variable
		key := h.envVarNameForToken(token.Type)
		return h.decoyManager.DeployEnvVar(ctx, key, token.Value)

	case TokenSSHKey:
		path := h.decoyManager.GetDecoyPath(token.Persona, token.Type)
		return h.decoyManager.DeployFile(ctx, path, token.Value)

	case TokenDNS:
		name := h.dnsNameForToken(token)
		return h.decoyManager.DeployDNSRecord(ctx, name, token.Value)

	default:
		return fmt.Errorf("no deployment method for token type: %s", token.Type)
	}
}

// envVarNameForToken returns appropriate environment variable name
func (h *HoneytokenInjector) envVarNameForToken(tokenType TokenType) string {
	names := map[TokenType]string{
		TokenAWSKey:       "AWS_SECRET_ACCESS_KEY",
		TokenAPIKey:       "API_KEY",
		TokenDBCredential: "DATABASE_URL",
		TokenGitToken:     "GITHUB_TOKEN",
	}
	if name, ok := names[tokenType]; ok {
		return name
	}
	return "HONEYTOKEN_" + strings.ToUpper(string(tokenType))
}

// dnsNameForToken generates a DNS record name
func (h *HoneytokenInjector) dnsNameForToken(token *Honeytoken) string {
	return fmt.Sprintf("honey-%s-%s.internal", token.SessionID[:8], token.ID[:8])
}

// OnHoneytokenAccessed handles honeytoken access events
func (h *HoneytokenInjector) OnHoneytokenAccessed(
	ctx context.Context,
	tokenID string,
	sessionID string,
) error {
	token, err := h.tokenStore.Get(ctx, tokenID)
	if err != nil {
		return err
	}

	now := time.Now()
	token.AccessedAt = &now
	token.AccessCount++

	if err := h.tokenStore.MarkAccessed(ctx, tokenID); err != nil {
		return err
	}

	if h.notificationSvc != nil {
		return h.notificationSvc.NotifyHoneytokenTouch(ctx, token, sessionID)
	}
	return nil
}

// ─── Token Generators ───

func generateAWSCredentials() string {
	accessKey := make([]byte, 20)
	rand.Read(accessKey)
	secretKey := make([]byte, 40)
	rand.Read(secretKey)
	return fmt.Sprintf("AKIA%s:%s", base64.StdEncoding.EncodeToString(accessKey)[:16],
		base64.StdEncoding.EncodeToString(secretKey))
}

func generateSSHKey() string {
	// Generate a fake RSA private key header
	keyData := make([]byte, 256)
	rand.Read(keyData)
	return fmt.Sprintf("-----BEGIN RSA PRIVATE KEY-----\n%s\n-----END RSA PRIVATE KEY-----",
		base64.StdEncoding.EncodeToString(keyData))
}

func generateAPIKey(persona Persona) string {
	prefixes := map[Persona]string{
		PersonaLegacyWin:   "sk-legacy-",
		PersonaK8sNode:     "sk-k8s-",
		PersonaCIRunner:    "sk-ci-",
		PersonaIoTGateway:  "sk-iot-",
		PersonaDevWS:       "sk-dev-",
		PersonaCloudFunc:   "sk-fn-",
	}
	prefix := prefixes[persona]
	if prefix == "" {
		prefix = "sk-"
	}
	keyBytes := make([]byte, 32)
	rand.Read(keyBytes)
	return prefix + hex.EncodeToString(keyBytes)
}

func generateDBCredential() string {
	hosts := []string{"db-prod.internal", "postgres.cluster.local", "mysql.master.internal"}
	user := "admin"
	passBytes := make([]byte, 16)
	rand.Read(passBytes)
	pass := hex.EncodeToString(passBytes)
	host := hosts[randInt(len(hosts))]
	return fmt.Sprintf("postgresql://%s:%s@%s:5432/production", user, pass, host)
}

func generateGitToken() string {
	tokenBytes := make([]byte, 20)
	rand.Read(tokenBytes)
	return "ghp_" + base64.URLEncoding.EncodeToString(tokenBytes)[:36]
}

func generateDNSToken(sessionID string) string {
	tokenBytes := make([]byte, 12)
	rand.Read(tokenBytes)
	return fmt.Sprintf("honey-%s-%s.exfil.attacker.com", sessionID[:8], hex.EncodeToString(tokenBytes))
}

// generateHoneyFileContent creates realistic honeyfile content per persona
func generateHoneyFileContent(persona Persona) string {
	watermark := fmt.Sprintf("HONEYTOKEN-%s-%d", persona, time.Now().Unix())

	templates := map[Persona]string{
		PersonaLegacyWin: `<?xml version="1.0"?>
<configuration>
  <connectionStrings>
    <add name="ProductionDB" connectionString="Server=sql-prod.internal;Database=ERP;User=sa;Password=P@ssw0rd2024;" />
  </connectionStrings>
  <appSettings>
    <add key="API_KEY" value="` + watermark + `" />
    <add key="AWS_ACCESS_KEY" value="AKIA` + watermark + `" />
  </appSettings>
</configuration>`,

		PersonaK8sNode: `apiVersion: v1
kind: Secret
metadata:
  name: prod-secrets
  namespace: production
type: Opaque
data:
  database-url: ` + base64.StdEncoding.EncodeToString([]byte("postgresql://admin:"+watermark+"@postgres:5432/prod")) + `
  api-key: ` + base64.StdEncoding.EncodeToString([]byte(watermark)) + `
  aws-secret: ` + base64.StdEncoding.EncodeToString([]byte("AKIA"+watermark)) + ` `,

		PersonaCIRunner: `#!/bin/bash
# CI/CD Deployment Script
export PROD_DB_URL="postgresql://deploy:` + watermark + `@db.prod.internal/app"
export AWS_ACCESS_KEY_ID="AKIA` + watermark + `"
export GITHUB_TOKEN="ghp_` + watermark + `"
export SLACK_WEBHOOK="https://hooks.slack.com/services/` + watermark + `"
kubectl apply -f k8s/production/`,

		PersonaIoTGateway: `{
  "device_id": "gw-` + watermark + `",
  "firmware_version": "2.4.1",
  "mqtt_broker": "mqtt.prod.internal",
  "credentials": {
    "username": "gateway",
    "password": "` + watermark + `",
    "api_key": "sk-iot-` + watermark + `"
  },
  "certificates": {
    "ca": "` + watermark + `-ca.pem",
    "client": "` + watermark + `-client.pem"
  }
}`,

		PersonaDevWS: `# Developer Environment Config
GITHUB_TOKEN=ghp_` + watermark + `
AWS_PROFILE=production
DATABASE_URL=postgresql://dev:` + watermark + `@db.dev.internal:5432/app
SSH_KEY_PATH=~/.ssh/id_rsa_` + watermark + `
NPM_TOKEN=` + watermark + `-npm
DOCKER_REGISTRY=registry.internal:5000`,

		PersonaCloudFunc: `{
  "function": "payment-processor",
  "runtime": "python3.11",
  "environment_variables": {
    "STRIPE_SECRET_KEY": "sk_live_` + watermark + `",
    "DATABASE_URL": "postgresql://fn:` + watermark + `@cloud-sql.internal/payments",
    "AWS_ACCESS_KEY_ID": "AKIA` + watermark + `",
    "WEBHOOK_SECRET": "` + watermark + `-whsec"
  },
  "vpc_connector": "projects/prod/locations/us-central1/connectors/vpc-prod"
}`,
	}

	if template, ok := templates[persona]; ok {
		return template
	}
	return fmt.Sprintf("# Honeytoken: %s\n%s", watermark, watermark)
}

func randInt(n int) int {
	b := make([]byte, 1)
	rand.Read(b)
	return int(b[0]) % n
}

// ─── Marshal/Unmarshal for storage ───

func (h *Honeytoken) MarshalJSON() ([]byte, error) {
	type Alias Honeytoken
	return json.Marshal(&struct {
		*Alias
		AccessedAt *string `json:"accessed_at,omitempty"`
	}{
		Alias:      (*Alias)(h),
		AccessedAt: timePtrToString(h.AccessedAt),
	})
}

func timePtrToString(t *time.Time) *string {
	if t == nil {
		return nil
	}
	s := t.Format(time.RFC3339)
	return &s
}
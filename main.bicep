// ════════════════════════════════════════════════════════════════
// Azure Bicep — Enterprise RAG Chatbot Infrastructure
// Resources: Container Apps + Azure OpenAI + Storage + Log Analytics
// ════════════════════════════════════════════════════════════════

param location string = resourceGroup().location
param appName string = 'nexus-rag'
param openAiSkuName string = 'S0'

// ── Log Analytics ─────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Container Apps Environment ────────────────────────────────
resource caEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ── Azure OpenAI ──────────────────────────────────────────────
resource openAi 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${appName}-aoai'
  location: location
  kind: 'OpenAI'
  sku: { name: openAiSkuName }
  properties: {
    customSubDomainName: '${appName}-aoai'
  }
}

resource embeddingDeploy 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAi
  name: 'text-embedding-ada-002'
  properties: {
    model: { format: 'OpenAI', name: 'text-embedding-ada-002', version: '2' }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: { name: 'Standard', capacity: 120 }
}

resource chatDeploy 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAi
  name: 'gpt-4o'
  dependsOn: [embeddingDeploy]
  properties: {
    model: { format: 'OpenAI', name: 'gpt-4o', version: '2024-05-13' }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: { name: 'Standard', capacity: 30 }
}

// ── Storage (for FAISS persistence) ──────────────────────────
resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${replace(appName,'-','')}store'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}

// ── Backend Container App ──────────────────────────────────────
resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${appName}-backend'
  location: location
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET','POST','DELETE','OPTIONS']
          allowedHeaders: ['*']
        }
      }
      secrets: [
        { name: 'aoai-key', value: openAi.listKeys().key1 }
      ]
    }
    template: {
      containers: [{
        name: 'backend'
        image: 'your-acr.azurecr.io/nexus-rag-backend:latest'
        resources: { cpu: json('1.0'), memory: '2Gi' }
        env: [
          { name: 'USE_AZURE_OPENAI', value: 'true' }
          { name: 'AZURE_OPENAI_ENDPOINT', value: openAi.properties.endpoint }
          { name: 'AZURE_OPENAI_API_KEY', secretRef: 'aoai-key' }
          { name: 'AZURE_OPENAI_API_VERSION', value: '2024-02-01' }
          { name: 'AZURE_EMBED_DEPLOYMENT', value: 'text-embedding-ada-002' }
          { name: 'AZURE_CHAT_DEPLOYMENT', value: 'gpt-4o' }
        ]
      }]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [{
          name: 'http-scaling'
          http: { metadata: { concurrentRequests: '20' } }
        }]
      }
    }
  }
}

// ── Frontend Container App ────────────────────────────────────
resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${appName}-frontend'
  location: location
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      ingress: { external: true, targetPort: 80 }
    }
    template: {
      containers: [{
        name: 'frontend'
        image: 'your-acr.azurecr.io/nexus-rag-frontend:latest'
        resources: { cpu: json('0.25'), memory: '0.5Gi' }
      }]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

// ── Outputs ────────────────────────────────────────────────────
output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output openAiEndpoint string = openAi.properties.endpoint

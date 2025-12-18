# Quickstart: Structured Data Vectorstore

**Feature**: 014-structured-data-ingestion
**Date**: 2025-12-18

This guide shows how to configure a vectorstore tool for semantic search over structured data (CSV, JSON).

## Prerequisites

- HoloDeck installed (`pip install holodeck-ai`)
- A structured data file (CSV or JSON)

## Basic Example: CSV Product Catalog

### 1. Prepare Your Data

Create `data/products.csv`:

```csv
id,title,description,category,price
1,Widget Pro,"Advanced widget with AI-powered features for professionals",Electronics,99.99
2,Super Gadget,"Multi-purpose gadget for daily home and office use",Electronics,149.99
3,Home Helper,"Smart home automation device with voice control",Smart Home,199.99
4,Office Suite,"Complete productivity software bundle",Software,299.99
5,Travel Kit,"Compact travel accessories for business travelers",Travel,79.99
```

### 2. Configure the Agent

Create `agent.yaml`:

```yaml
name: product-assistant
description: AI assistant for product recommendations

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7

instructions:
  inline: |
    You are a helpful product assistant. Use the product_search tool
    to find relevant products based on user queries. Always include
    the product title, description, and price in your responses.

tools:
  - name: product_search
    type: vectorstore
    description: Search product catalog by semantic similarity
    source: data/products.csv
    id_field: id
    vector_field: description
    meta_fields: [title, category, price]
```

### 3. Test the Configuration

```bash
# Validate configuration
holodeck test agent.yaml --dry-run

# Interactive chat
holodeck chat agent.yaml
```

## Advanced Example: JSON FAQ with Multiple Fields

### 1. Prepare Your Data

Create `data/faqs.json`:

```json
[
  {
    "faq_id": "FAQ001",
    "question": "How do I reset my password?",
    "answer": "Go to Settings > Security > Reset Password. You will receive an email with a reset link.",
    "category": "Account"
  },
  {
    "faq_id": "FAQ002",
    "question": "What payment methods do you accept?",
    "answer": "We accept Visa, Mastercard, American Express, PayPal, and bank transfers.",
    "category": "Billing"
  },
  {
    "faq_id": "FAQ003",
    "question": "How can I contact support?",
    "answer": "You can reach us via email at support@example.com or call 1-800-SUPPORT.",
    "category": "Support"
  }
]
```

### 2. Configure with Multiple Vector Fields

```yaml
tools:
  - name: faq_search
    type: vectorstore
    description: Search FAQ knowledge base
    source: data/faqs.json
    id_field: faq_id
    vector_fields: [question, answer]  # Combine both fields for embedding
    field_separator: "\n\n"            # Double newline between fields
    meta_fields: [category]
```

This configuration embeds both question and answer together, making the search more comprehensive.

## Configuration Reference

### Required Fields (Structured Mode)

| Field | Type | Description |
|-------|------|-------------|
| `id_field` | string | Field containing unique record identifier |
| `vector_field` | string | Single field to embed for search |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vector_fields` | list | - | Multiple fields to embed (alternative to `vector_field`) |
| `meta_fields` | list | all fields | Fields to include in search results |
| `field_separator` | string | `\n` | Separator when joining multiple vector fields |
| `delimiter` | string | auto-detect | CSV delimiter (comma, semicolon, tab) |
| `record_path` | string | - | JSON path to array of records |

### Structured vs Unstructured Mode

| Feature | Structured Mode | Unstructured Mode |
|---------|-----------------|-------------------|
| Triggered by | `id_field` + `vector_field` set | Neither set |
| Use case | Tabular data (CSV, JSON) | Documents (PDF, MD, TXT) |
| Embedding | Specific field(s) | Full document content |
| Chunking | One embedding per record | Multiple chunks per document |

## Troubleshooting

### "Field 'X' not found" Error

The specified field doesn't exist in your data source. Check:
- Spelling matches exactly (case-sensitive)
- Field exists in all records
- For nested JSON: use dot notation (e.g., `details.description`)

### Empty Search Results

- Verify the `vector_field` contains meaningful text content
- Check that records aren't being skipped due to empty values
- Try increasing `top_k` for more results

### CSV Parsing Issues

- Specify `delimiter` explicitly if auto-detection fails
- Ensure consistent quoting for fields with special characters
- Check file encoding (UTF-8 recommended)

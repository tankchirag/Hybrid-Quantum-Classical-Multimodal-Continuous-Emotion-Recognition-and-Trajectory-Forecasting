```mermaid
flowchart TD
RawModalities --> FeatureExtractors
FeatureExtractors --> Tokenizer
Tokenizer --> TokenProjection
TokenProjection --> SelfAttention
SelfAttention --> TokenPooling
TokenPooling --> TransformerEncoder
TransformerEncoder --> TransformerDecoder
TransformerDecoder --> RecognitionHead
RecognitionHead --> ForecastingHead
```
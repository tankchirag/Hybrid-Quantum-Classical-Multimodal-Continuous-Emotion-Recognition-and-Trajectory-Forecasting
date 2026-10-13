```mermaid
flowchart TD
RawModalities --> FeatureExtractors
FeatureExtractors --> Tokenizer
Tokenizer --> Projection128To64
Projection128To64 --> ConcatenationFusion
ConcatenationFusion --> TransformerEncoder
TransformerEncoder --> TransformerDecoder
TransformerDecoder --> RecognitionHead
RecognitionHead --> ForecastingHead
```
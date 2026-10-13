```mermaid
flowchart TD
RawModalities --> FeatureExtractors
FeatureExtractors --> SourceStack
SourceStack --> Projection128ToQ
Projection128ToQ --> QPQC
QPQC --> KPQC
KPQC --> VPQC
VPQC --> GaussianAttention
GaussianAttention --> SourcePooling
SourcePooling --> TransformerEncoder
TransformerEncoder --> TransformerDecoder
TransformerDecoder --> RecognitionHead
RecognitionHead --> ForecastingHead
```
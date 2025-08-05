import os
import sys
from pathlib import Path
from getpass import getpass
import yaml

# Standalone function to load environment variables without importing app.py
def load_env_vars():
    """Load environment variables from env.yaml file."""
    env_vars = {}
    try:
        # Look for env.yaml in the project root (two levels up from this file)
        env_file_path = Path(__file__).parent.parent.parent / 'env.yaml'
        
        if env_file_path.exists():
            with open(env_file_path, 'r') as file:
                content = file.read()
                # Simple parsing for each key in the YAML file
                for line in content.splitlines():
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('\'"')
                        if value:  # Only add non-empty values
                            env_vars[key] = value
                            # Also set as environment variable for other modules to access
                            os.environ[key.upper()] = value
        return env_vars
    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return {}

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Advanced chunking and retrieval imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Try different import paths for RAGAS evolutions (version compatibility)
try:
    from ragas.testset.evolutions import simple, reasoning, multi_context
except ImportError:
    try:
        # Alternative import path for different RAGAS versions
        from ragas.testset.generator import simple, reasoning, multi_context
    except ImportError:
        # Fallback - define basic distributions
        print("⚠️  Using basic question distributions (evolutions import failed)")
        simple = "simple"
        reasoning = "reasoning" 
        multi_context = "multi_context"

import pandas as pd

class RAGASSDGGenerator:
    """RAGAS Synthetic Data Generator for creating evaluation datasets."""
    
    def __init__(self, data_path="../data/", model_name="gpt-4o-mini"):
        """
        Initialize the SDG Generator.
        
        Args:
            data_path (str): Path to the directory containing PDF documents
            model_name (str): OpenAI model to use for generation
        """
        self.data_path = data_path
        self.model_name = model_name
        self.docs = None
        self.generator = None
        self._setup_api_key()
        self._setup_generator()
    
    def _setup_api_key(self):
        """Load and set up the OpenAI API key."""
        try:
            env_vars = load_env_vars()
            api_key = env_vars.get('openai_api_key', '')
            if not api_key:
                api_key = getpass("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
        except Exception as e:
            print(f"Error loading API key: {e}")
            api_key = getpass("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
    
    def _setup_generator(self):
        """Set up the RAGAS test generator with LLM and embeddings."""
        try:
            generator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
            generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            
            self.generator = TestsetGenerator(
                llm=generator_llm, 
                embedding_model=generator_embeddings
            )
            print(f"✅ RAGAS generator initialized with {self.model_name}")
        except Exception as e:
            print(f"❌ Error setting up generator: {e}")
            raise
    
    def load_documents(self, max_docs=None, max_file_size_mb=30, chunking_strategy="recursive", chunk_size=1000, chunk_overlap=200):
        """
        Load PDF documents with advanced chunking strategies.
        
        Args:
            max_docs (int, optional): Maximum number of documents to load
            max_file_size_mb (int): Maximum file size in MB to avoid vector DB limits
            chunking_strategy (str): Strategy for chunking ("recursive", "token", "semantic")
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        try:
            if not os.path.exists(self.data_path):
                print(f"❌ Data path {self.data_path} does not exist")
                return False
            
            # Get all PDF files and filter by size
            pdf_files = []
            data_path = Path(self.data_path)
            
            for pdf_file in data_path.glob("*.pdf"):
                file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
                if file_size_mb <= max_file_size_mb:
                    pdf_files.append(str(pdf_file))
                else:
                    print(f"⚠️  Skipping {pdf_file.name} ({file_size_mb:.1f}MB) - exceeds {max_file_size_mb}MB limit")
            
            if not pdf_files:
                print("❌ No suitable PDF files found within size limit")
                return False
            
            print(f"📁 Found {len(pdf_files)} PDF files within size limit")
            
            # Load documents one by one to handle individual errors
            all_docs = []
            for pdf_file in pdf_files:
                try:
                    loader = PyMuPDFLoader(pdf_file)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"✅ Loaded {len(docs)} pages from {Path(pdf_file).name}")
                except Exception as e:
                    print(f"⚠️  Error loading {Path(pdf_file).name}: {e}")
                    continue
            
            if max_docs and len(all_docs) > max_docs:
                all_docs = all_docs[:max_docs]
                print(f"📄 Limited to first {max_docs} documents")
            
            # Apply advanced chunking strategy
            print(f"🔄 Applying {chunking_strategy} chunking strategy...")
            chunked_docs = self._apply_chunking_strategy(
                all_docs, 
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            self.docs = chunked_docs
            print(f"✅ Total chunks created: {len(self.docs)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return False
    
    def _apply_chunking_strategy(self, docs, strategy="recursive", chunk_size=1000, chunk_overlap=200):
        """
        Apply advanced chunking strategies to documents.
        
        Args:
            docs: List of documents to chunk
            strategy: Chunking strategy to use
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of chunked documents
        """
        try:
            if strategy == "recursive":
                # Best for general text - splits on paragraphs, sentences, then words
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            elif strategy == "token":
                # Token-based splitting for more precise control
                text_splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                # Default to recursive
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            # Split documents and filter out very short chunks
            chunked_docs = text_splitter.split_documents(docs)
            
            # Filter out chunks that are too short (likely headers/footers)
            filtered_chunks = []
            for chunk in chunked_docs:
                # Skip chunks that are mostly whitespace or too short
                content = chunk.page_content.strip()
                if len(content) > 50 and not self._is_likely_header_footer(content):
                    filtered_chunks.append(chunk)
            
            print(f"📊 Chunking results: {len(chunked_docs)} total → {len(filtered_chunks)} filtered chunks")
            return filtered_chunks
            
        except Exception as e:
            print(f"❌ Error in chunking strategy: {e}")
            return docs  # Return original docs if chunking fails
    
    def _is_likely_header_footer(self, text):
        """
        Identify and filter out likely headers, footers, and table of contents.
        
        Args:
            text: Text content to check
        
        Returns:
            bool: True if likely header/footer/TOC
        """
        text_lower = text.lower()
        
        # Common header/footer patterns
        header_footer_patterns = [
            "copyright", "all rights reserved", "page", "table of contents",
            "level 2 training guide", "crossfit, inc", "v5.1-", "workbook |"
        ]
        
        # TOC patterns
        toc_patterns = [
            "...", "summary sheet:", "course overview", "preparation for",
            "learning objectives", "schedule", "day 1", "day 2"
        ]
        
        # Check if text is mostly dots (TOC formatting)
        dot_ratio = text.count('.') / len(text) if len(text) > 0 else 0
        
        # Check patterns
        for pattern in header_footer_patterns + toc_patterns:
            if pattern in text_lower:
                return True
        
        # Check if mostly dots (TOC)
        if dot_ratio > 0.3:
            return True
        
        # Check if very short and contains numbers (page numbers)
        if len(text) < 100 and any(char.isdigit() for char in text):
            return True
        
        return False
    
    def create_advanced_retriever(self, retriever_type="ensemble", k=5):
        """
        Create advanced retriever for better context retrieval.
        
        Args:
            retriever_type: Type of retriever ("bm25", "ensemble", "compression", "multi_query", "parent")
            k: Number of documents to retrieve
        
        Returns:
            Advanced retriever object
        """
        if not self.docs:
            print("❌ No documents loaded. Call load_documents() first.")
            return None
        
        try:
            print(f"🔄 Creating {retriever_type} retriever...")
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(self.docs, embeddings)
            
            if retriever_type == "bm25":
                # BM25 (keyword-based) retriever
                retriever = BM25Retriever.from_documents(self.docs)
                retriever.k = k
                
            elif retriever_type == "ensemble":
                # Ensemble of semantic + keyword retrieval
                semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                bm25_retriever = BM25Retriever.from_documents(self.docs)
                bm25_retriever.k = k
                
                retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, bm25_retriever],
                    weights=[0.6, 0.4]  # Favor semantic over keyword
                )
                
            elif retriever_type == "compression":
                # Compression retriever to filter irrelevant context
                base_retriever = vectorstore.as_retriever(search_kwargs={"k": k*2})
                compressor = LLMChainExtractor.from_llm(ChatOpenAI(model=self.model_name))
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                
            elif retriever_type == "multi_query":
                # Multi-query retriever for better question understanding
                base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=ChatOpenAI(model=self.model_name)
                )
                
            elif retriever_type == "parent":
                # Parent document retriever for better context
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                
                store = InMemoryStore()
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
                retriever.add_documents(self.docs)
                
            else:
                # Default to semantic retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            
            print(f"✅ {retriever_type.title()} retriever created successfully")
            return retriever
            
        except Exception as e:
            print(f"❌ Error creating {retriever_type} retriever: {e}")
            return None
    
    def generate_synthetic_dataset(self, testset_size=9, distributions=None):
        """
        Generate synthetic test dataset using RAGAS.
        
        Args:
            testset_size (int): Number of test cases to generate
            distributions (dict): Distribution of question types (ignored in RAGAS 0.2.10)
        
        Returns:
            Dataset: Generated synthetic dataset
        """
        if not self.docs:
            print("❌ No documents loaded. Please call load_documents() first.")
            return None
        
        if not self.generator:
            print("❌ Generator not initialized.")
            return None
        
        try:
            print(f"🔄 Generating {testset_size} synthetic test cases...")
            
            # RAGAS 0.2.10 API - use basic generation without distributions parameter
            dataset = self.generator.generate_with_langchain_docs(
                self.docs, 
                testset_size=testset_size
            )
            
            print(f"✅ Generated {len(dataset)} test cases successfully!")
            return dataset
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            
            # Try with even fewer documents if there's a memory/processing issue
            if len(self.docs) > 10:
                print("💡 Trying with fewer documents...")
                try:
                    dataset = self.generator.generate_with_langchain_docs(
                        self.docs[:10], 
                        testset_size=min(testset_size, 5)
                    )
                    print(f"✅ Generated {len(dataset)} test cases with reduced dataset!")
                    return dataset
                except Exception as e2:
                    print(f"❌ Reduced generation also failed: {e2}")
            
            return None
    
    def save_dataset(self, dataset, filename="synthetic_dataset.csv"):
        """
        Save the generated dataset to a CSV file.
        
        Args:
            dataset: The generated dataset
            filename (str): Output filename
        """
        if dataset is None:
            print("❌ No dataset to save.")
            return False
        
        try:
            df = dataset.to_pandas()
            output_path = os.path.join(os.path.dirname(__file__), filename)
            df.to_csv(output_path, index=False)
            print(f"✅ Dataset saved to {output_path}")
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def preview_dataset(self, dataset, num_samples=3):
        """
        Preview the generated dataset.
        
        Args:
            dataset: The generated dataset
            num_samples (int): Number of samples to display
        """
        if dataset is None:
            print("❌ No dataset to preview.")
            return
        
        try:
            df = dataset.to_pandas()
            print(f"\n📊 Dataset Preview ({num_samples} samples):")
            print("=" * 50)
            
            for i in range(min(num_samples, len(df))):
                print(f"\n🔍 Sample {i+1}:")
                print(f"Question: {df.iloc[i]['user_input']}")
                print(f"Answer: {df.iloc[i]['reference']}")
                if 'reference_contexts' in df.columns:
                    contexts = df.iloc[i]['reference_contexts']
                    if isinstance(contexts, list) and contexts:
                        print(f"Context: {contexts[0][:200]}...")
                print("-" * 30)
        except Exception as e:
            print(f"❌ Error previewing dataset: {e}")

    def evaluate_with_ragas(self, dataset_path=None, dataset=None, sample_size=None):
        """
        Evaluate the synthetic dataset using RAGAS metrics.
        
        Args:
            dataset_path (str): Path to CSV file containing the dataset
            dataset (Dataset): RAGAS dataset object
            sample_size (int): Number of samples to evaluate (for faster testing)
        
        Returns:
            dict: Evaluation results
        """
        try:
            # Load dataset if path provided
            if dataset_path:
                print(f"📁 Loading dataset from {dataset_path}")
                df = pd.read_csv(dataset_path)
            elif dataset:
                df = dataset.to_pandas()
            else:
                print("❌ No dataset provided for evaluation")
                return None
            
            # Sample dataset if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"📊 Using {sample_size} samples for evaluation")
            
            print(f"🔄 Evaluating {len(df)} samples with RAGAS metrics...")
            
            # Prepare data for RAGAS evaluation
            # RAGAS expects specific column names
            eval_data = {
                'question': df['user_input'].tolist(),
                'answer': df['reference'].tolist(),
                'contexts': df['reference_contexts'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                ).tolist(),
                'ground_truth': df['reference'].tolist()  # Using reference as ground truth
            }
            
            # Create RAGAS dataset
            ragas_dataset = Dataset.from_dict(eval_data)
            
            # Define metrics to evaluate
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
            
            print("📊 Running RAGAS evaluation...")
            print("   Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
            
            # Run evaluation
            result = evaluate(
                dataset=ragas_dataset,
                metrics=metrics,
                llm=self.generator.llm,  # Use the wrapper directly
                embeddings=self.generator.embedding_model  # Use the wrapper directly
            )
            
            print("\n✅ RAGAS Evaluation Complete!")
            print("=" * 50)
            
            # Display results - handle EvaluationResult object
            try:
                # Try to access results as DataFrame first (RAGAS 0.2.10 format)
                if hasattr(result, 'to_pandas'):
                    result_df = result.to_pandas()
                    print("📊 Evaluation Results:")
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                        if metric in result_df.columns:
                            avg_score = result_df[metric].mean()
                            print(f"📈 {metric.replace('_', ' ').title()}: {avg_score:.4f}")
                
                # Try dictionary-style access as fallback
                elif hasattr(result, 'items'):
                    for metric_name, score in result.items():
                        if isinstance(score, (int, float)):
                            print(f"📈 {metric_name.replace('_', ' ').title()}: {score:.4f}")
                
                # Try direct attribute access
                else:
                    metrics_to_check = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
                    for metric in metrics_to_check:
                        if hasattr(result, metric):
                            score = getattr(result, metric)
                            if isinstance(score, (int, float)):
                                print(f"📈 {metric.replace('_', ' ').title()}: {score:.4f}")
                            elif hasattr(score, 'mean'):  # If it's an array-like object
                                print(f"📈 {metric.replace('_', ' ').title()}: {score.mean():.4f}")
                
            except Exception as display_error:
                print(f"⚠️  Could not display detailed results: {display_error}")
                print(f"📋 Raw result type: {type(result)}")
                print(f"📋 Available attributes: {dir(result)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during RAGAS evaluation: {e}")
            print("💡 Make sure your dataset has the correct column names:")
            print("   - user_input (questions)")
            print("   - reference (answers)")
            print("   - reference_contexts (contexts)")
            return None
    
    def save_evaluation_results(self, results, filename="ragas_evaluation_results.json"):
        """
        Save RAGAS evaluation results to a JSON file.
        
        Args:
            results: Evaluation results from RAGAS (EvaluationResult object or dict)
            filename (str): Output filename
        """
        if results is None:
            print("❌ No results to save.")
            return False
        
        try:
            import json
            output_path = os.path.join(os.path.dirname(__file__), filename)
            
            # Handle different result formats from RAGAS 0.2.10
            json_results = {}
            
            try:
                # Try to convert to pandas DataFrame first (RAGAS 0.2.10 format)
                if hasattr(results, 'to_pandas'):
                    result_df = results.to_pandas()
                    # Calculate mean scores for each metric
                    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                        if metric in result_df.columns:
                            json_results[metric] = float(result_df[metric].mean())
                    
                    # Also save individual scores
                    json_results['individual_scores'] = result_df.to_dict('records')
                
                # Fallback to dictionary-style access
                elif hasattr(results, 'items'):
                    for key, value in results.items():
                        if isinstance(value, (int, float, str, bool, list, dict)):
                            json_results[key] = value
                        else:
                            json_results[key] = str(value)
                
                # Fallback to attribute access
                else:
                    metrics_to_check = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
                    for metric in metrics_to_check:
                        if hasattr(results, metric):
                            score = getattr(results, metric)
                            if isinstance(score, (int, float)):
                                json_results[metric] = float(score)
                            elif hasattr(score, 'mean'):  # If it's an array-like object
                                json_results[metric] = float(score.mean())
                            else:
                                json_results[metric] = str(score)
                
                # Add metadata
                json_results['evaluation_metadata'] = {
                    'result_type': str(type(results)),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'ragas_version': '0.2.10'
                }
                
            except Exception as parse_error:
                print(f"⚠️  Could not parse results, saving raw format: {parse_error}")
                json_results = {
                    'raw_result': str(results),
                    'result_type': str(type(results)),
                    'error': str(parse_error)
                }
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"✅ Evaluation results saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

def main():
    """Main function to demonstrate SDG usage."""
    print("🚀 Starting RAGAS Synthetic Data Generation...")
    
    # Initialize the SDG generator
    sdg = RAGASSDGGenerator(data_path="../data/")
    
    # Load documents
    if not sdg.load_documents(max_docs=20):
        print("❌ Failed to load documents. Please check the data path.")
        return
    
    # Generate synthetic dataset
    dataset = sdg.generate_synthetic_dataset(testset_size=9)
    
    if dataset:
        # Preview the dataset
        sdg.preview_dataset(dataset)
        
        # Save the dataset
        sdg.save_dataset(dataset, "crossfit_synthetic_dataset.csv")
        
        # Evaluate the dataset with RAGAS
        results = sdg.evaluate_with_ragas(dataset=dataset)
        
        # Save the evaluation results
        if results:
            sdg.save_evaluation_results(results)
        
        print("\n✅ SDG process completed successfully!")
    else:
        print("❌ Failed to generate synthetic dataset.")

if __name__ == "__main__":
    main()
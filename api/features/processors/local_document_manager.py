import os
import uuid
from typing import List, Dict, Any
from api.features.processors.document_processor import DocumentProcessor
from api.features.store.vector_store import QdrantVectorStore

class LocalDocumentManager:
    """Manages local PDFs in the api/data directory for RAG processing"""
    
    def __init__(self, data_dir: str = None, collection_name: str = "documents"):
        # Default to api/data directory relative to this file
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_dir = os.path.dirname(os.path.dirname(current_dir))
            self.data_dir = os.path.join(api_dir, "data")
        else:
            self.data_dir = data_dir
            
        self.document_processor = DocumentProcessor(collection_name=collection_name)
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def scan_local_pdfs(self) -> List[str]:
        """Scan the data directory for PDF files"""
        pdf_files = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(self.data_dir, filename))
        return pdf_files
    
    def get_local_pdf_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all local PDFs"""
        pdf_files = self.scan_local_pdfs()
        metadata_list = []
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            file_size = os.path.getsize(pdf_path)
            
            # Generate a consistent file_id based on filename
            file_id = f"local_{filename.replace('.pdf', '').replace(' ', '_').lower()}"
            
            metadata_list.append({
                "file_id": file_id,
                "filename": filename,
                "file_path": pdf_path,
                "file_size": file_size,
                "source": "local",
                "is_local": True
            })
        
        return metadata_list
    
    def is_pdf_indexed(self, file_id: str) -> bool:
        """Check if a PDF is already indexed in the vector store"""
        try:
            existing_pdfs = self.vector_store.get_all_pdf_metadata()
            return any(pdf.get("file_id") == file_id for pdf in existing_pdfs)
        except Exception as e:
            print(f"Error checking if PDF is indexed: {e}")
            return False
    
    def index_local_pdf(self, pdf_path: str, file_id: str = None) -> Dict[str, Any]:
        """Index a single local PDF"""
        filename = os.path.basename(pdf_path)
        
        # Generate file_id if not provided
        if file_id is None:
            file_id = f"local_{filename.replace('.pdf', '').replace(' ', '_').lower()}"
        
        print(f"Indexing local PDF: {filename} with file_id: {file_id}")
        
        try:
            result = self.document_processor.process_pdf(
                pdf_path,
                custom_filename=filename,
                custom_file_id=file_id
            )
            
            # Mark as local document
            result["source"] = "local"
            result["is_local"] = True
            
            return result
        except Exception as e:
            print(f"Error indexing local PDF {filename}: {e}")
            raise
    
    def index_all_local_pdfs(self, force_reindex: bool = False) -> List[Dict[str, Any]]:
        """Index all local PDFs that aren't already indexed"""
        pdf_metadata = self.get_local_pdf_metadata()
        results = []
        
        for pdf_info in pdf_metadata:
            file_id = pdf_info["file_id"]
            pdf_path = pdf_info["file_path"]
            filename = pdf_info["filename"]
            
            # Skip if already indexed (unless force_reindex is True)
            if not force_reindex and self.is_pdf_indexed(file_id):
                print(f"PDF {filename} already indexed, skipping...")
                results.append({
                    "filename": filename,
                    "file_id": file_id,
                    "status": "already_indexed",
                    "source": "local",
                    "is_local": True
                })
                continue
            
            try:
                result = self.index_local_pdf(pdf_path, file_id)
                results.append(result)
                print(f"Successfully indexed local PDF: {filename}")
            except Exception as e:
                print(f"Failed to index local PDF {filename}: {e}")
                results.append({
                    "filename": filename,
                    "file_id": file_id,
                    "status": "error",
                    "error": str(e),
                    "source": "local",
                    "is_local": True
                })
        
        return results
    
    def get_combined_pdf_list(self) -> List[Dict[str, Any]]:
        """Get a combined list of local and uploaded PDFs"""
        # Get local PDFs
        local_pdfs = self.get_local_pdf_metadata()
        
        # Get uploaded PDFs from vector store
        try:
            uploaded_pdfs = self.vector_store.get_all_pdf_metadata()
        except Exception as e:
            print(f"Error getting uploaded PDFs: {e}")
            uploaded_pdfs = []
        
        # Mark uploaded PDFs
        for pdf in uploaded_pdfs:
            if not pdf.get("is_local", False):
                pdf["source"] = "uploaded"
                pdf["is_local"] = False
        
        # Combine and deduplicate by file_id
        all_pdfs = {}
        
        # Add local PDFs first
        for pdf in local_pdfs:
            all_pdfs[pdf["file_id"]] = pdf
        
        # Add uploaded PDFs (won't overwrite local ones due to different file_id patterns)
        for pdf in uploaded_pdfs:
            file_id = pdf.get("file_id")
            if file_id and file_id not in all_pdfs:
                all_pdfs[file_id] = pdf
        
        return list(all_pdfs.values())

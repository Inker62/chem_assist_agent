"""RDKit 分子计算工具——SMILES 标准化、分子描述符计算、2D 结构图生成"""
import os
import json
import logging
from typing import Type

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "local_data", "mol_images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def compute_descriptors(smiles: str) -> dict:
    """从 SMILES 计算分子描述符，返回 dict"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"无法解析 SMILES: {smiles}"}

    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "num_atoms": mol.GetNumAtoms(),
        "num_rings": Descriptors.RingCount(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
    }


def generate_2d_image(smiles: str, filename: str = None) -> str | None:
    """生成 2D 分子结构图，返回图片路径，失败返回 None"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    if filename is None:
        safe_name = smiles.replace("/", "_").replace("\\", "_")[:50]
        filename = f"{safe_name}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    Draw.MolToFile(mol, filepath, size=(400, 300))
    return filepath


class ChemCalcInput(BaseModel):
    smiles: str = Field(description="化合物的 SMILES 结构式，例如 'CC(=O)Oc1ccccc1C(=O)O'")
    compute_2d: bool = Field(default=True, description="是否生成 2D 结构图")


class ChemCalcTool(BaseTool):
    """RDKit 分子计算与结构图生成工具"""
    name: str = "ChemCalc"
    description: str = (
        "当得到化合物的 SMILES 结构式后，使用此工具计算分子量、分子式、LogP 等描述符，并可生成 2D 结构图。"
        "输入：SMILES 字符串和可选的 compute_2d 标志。"
        "返回：分子描述符 JSON，包含规范化 SMILES、分子式、分子量、LogP 等信息，以及 2D 结构图路径。"
    )
    args_schema: Type[BaseModel] = ChemCalcInput

    def _run(self, smiles: str, compute_2d: bool = True) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"无法解析 SMILES 字符串: {smiles}"}, ensure_ascii=False)

        result = compute_descriptors(smiles)
        logger.info("计算完成: %s, MW=%.2f, LogP=%.2f",
                     result.get("smiles", "?"),
                     result.get("molecular_weight", 0),
                     result.get("logp", 0))

        if compute_2d:
            image_path = generate_2d_image(smiles)
            if image_path:
                result["image_path"] = image_path

        return json.dumps(result, indent=2, ensure_ascii=False)


def extract_mol_images(messages: list) -> list:
    """从消息列表中提取 ChemCalc 生成的 2D 结构图路径，返回 [(path, smiles), ...]"""
    import json as _json
    images = []
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            try:
                data = _json.loads(msg.content)
                if isinstance(data, dict) and "image_path" in data:
                    img_path = data["image_path"]
                    if os.path.exists(img_path):
                        images.append((img_path, data.get("smiles", "")))
            except (_json.JSONDecodeError, TypeError):
                pass
    return images

"""RDKit 分子计算工具——SMILES 标准化、分子描述符计算、2D 结构图生成、结构标注"""
import os
import json
import logging
from typing import Type

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "local_data", "mol_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==================== 官能团 SMARTS 模式库 ====================
FUNCTIONAL_GROUPS = {
    "羧酸": {"smarts": "[CX3](=O)[OX2H1]", "description": "羧基 -COOH"},
    "酯基": {"smarts": "[CX3](=O)[OX2H0][#6]", "description": "酯基 -COOR"},
    "酰胺": {"smarts": "[CX3](=O)[NX3]", "description": "酰胺 -CONR₂"},
    "伯胺": {"smarts": "[NX3H2;!$(NC=O)]", "description": "伯胺 -NH₂"},
    "仲胺": {"smarts": "[NX3H1;!$(NC=O)]([#6])[#6]", "description": "仲胺 -NHR"},
    "叔胺": {"smarts": "[NX3H0;!$(NC=O)]([#6])([#6])[#6]", "description": "叔胺 -NR₃"},
    "醇羟基": {"smarts": "[OX2H1][CX4;!$(C(=O))]", "description": "醇羟基 -OH"},
    "酚羟基": {"smarts": "[OX2H1][c]", "description": "酚羟基 Ar-OH"},
    "醛基": {"smarts": "[CX3H1](=O)[#6]", "description": "醛基 -CHO"},
    "酮": {"smarts": "[#6][CX3](=O)[#6]", "description": "酮羰基 >C=O"},
    "醚": {"smarts": "[OX2H0]([#6])[#6]", "description": "醚键 -O-"},
    "氰基": {"smarts": "[CX2]#[NX1]", "description": "氰基 -CN"},
    "硝基": {"smarts": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][#6]", "description": "硝基 -NO₂"},
    "磺酸": {"smarts": "[SX4](=O)(=O)[OX2H1]", "description": "磺酸基 -SO₃H"},
    "氟": {"smarts": "[F]", "description": "氟取代基 -F"},
    "氯": {"smarts": "[Cl]", "description": "氯取代基 -Cl"},
    "溴": {"smarts": "[Br]", "description": "溴取代基 -Br"},
    "碘": {"smarts": "[I]", "description": "碘取代基 -I"},
    "碳碳双键": {"smarts": "[CX3]=[CX3]", "description": "C=C 双键"},
    "碳碳三键": {"smarts": "[CX2]#[CX2]", "description": "C≡C 三键"},
    "苯环": {"smarts": "c1ccccc1", "description": "苯环"},
    "环氧": {"smarts": "C1OC1", "description": "环氧三元环"},
}


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


def annotate_structure(smiles: str) -> dict:
    """分子结构标注——手性中心、E/Z 异构、官能团、核心骨架、反应活性位点"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"无法解析 SMILES: {smiles}"}

    result = {}
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    result["smiles"] = canonical_smiles

    # 1. 手性中心检测
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if chiral_centers:
        chiral_list = []
        for idx, tag in chiral_centers:
            atom = mol.GetAtomWithIdx(idx)
            chiral_list.append({
                "atom_index": idx,
                "stereo": tag,
                "symbol": atom.GetSymbol(),
            })
        result["chiral_centers"] = chiral_list

    # 2. E/Z 异构检测
    ez_bonds = []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            ez_bonds.append({
                "bond_index": bond.GetIdx(),
                "type": "E (trans/反式)",
                "atoms": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
            })
        elif stereo == Chem.BondStereo.STEREOZ:
            ez_bonds.append({
                "bond_index": bond.GetIdx(),
                "type": "Z (cis/顺式)",
                "atoms": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
            })
    if ez_bonds:
        result["ez_isomers"] = ez_bonds

    # 3. 官能团识别
    detected_groups = []
    for name, info in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(info["smarts"])
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            detected_groups.append({
                "name": name,
                "description": info["description"],
                "count": len(matches),
            })
    if detected_groups:
        result["functional_groups"] = detected_groups

    # 4. Murcko 骨架提取
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold and scaffold.GetNumAtoms() > 0:
            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
            result["murcko_scaffold"] = scaffold_smiles
    except Exception:
        pass

    # 5. 反应活性位点概览
    reactive_sites = []
    # 亲电位点（羰基碳）
    electrophile_pattern = Chem.MolFromSmarts("[CX3](=O)")
    if electrophile_pattern:
        for match in mol.GetSubstructMatches(electrophile_pattern):
            reactive_sites.append({
                "atom_index": match[0],
                "type": "亲电位点 (羰基碳)",
                "reactivity": "易受亲核进攻",
            })
    # 亲核位点——胺
    for smarts, label in [
        ("[NX3H2;!$(NC=O)]", "亲核位点 (伯胺 NH₂)"),
        ("[NX3H1;!$(NC=O)]", "亲核位点 (仲胺 NH)"),
        ("[OX2H1][CX4;!$(C(=O))]", "亲核位点 (醇羟基 OH)"),
    ]:
        nuc_pattern = Chem.MolFromSmarts(smarts)
        if nuc_pattern:
            for match in mol.GetSubstructMatches(nuc_pattern):
                reactive_sites.append({
                    "atom_index": match[0],
                    "type": label,
                    "reactivity": "可作为亲核试剂",
                })
    if reactive_sites:
        result["reactive_sites"] = reactive_sites

    return result


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


class StructureAnnotationInput(BaseModel):
    smiles: str = Field(description="化合物的 SMILES 结构式")


class StructureAnnotationTool(BaseTool):
    """分子结构标注工具——识别手性中心、E/Z 异构、官能团、核心骨架和反应活性位点"""
    name: str = "StructureAnnotator"
    description: str = (
        "当需要深入分析分子结构时使用此工具。功能包括：\n"
        "1. 手性中心检测 (R/S 构型)\n"
        "2. E/Z (顺反) 异构识别\n"
        "3. 官能团自动标注（羧酸、酯、酰胺、胺、醇、醛、酮等 20+ 种）\n"
        "4. Murcko 核心骨架提取\n"
        "5. 反应活性位点识别（亲电/亲核位点）\n"
        "输入：SMILES 字符串。\n"
        "返回：含手性中心、官能团列表、骨架 SMILES、活性位点等信息的 JSON。"
    )
    args_schema: Type[BaseModel] = StructureAnnotationInput

    def _run(self, smiles: str) -> str:
        result = annotate_structure(smiles)
        logger.info("结构标注完成: %s, 手性中心=%d, 官能团=%d",
                     result.get("smiles", "?"),
                     len(result.get("chiral_centers", [])),
                     len(result.get("functional_groups", [])))
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

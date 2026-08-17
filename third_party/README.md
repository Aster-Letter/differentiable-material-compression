# Third-party

外部依赖默认通过安装说明、固定 commit 或 Git submodule 获取，不直接在第三方目录开发。重点参考：

- https://github.com/NVlabs/nvdiffrast
- https://github.com/NVlabs/nvdiffmodeling

推荐方案：现代环境直接安装固定 commit 的 nvdiffrast；nvdiffmodeling 固定为只读源码参考，选择性迁移 PBR/材质思路并保留许可证与出处。详见 [`../docs/references/nvlabs-integration.md`](../docs/references/nvlabs-integration.md)。

本机忽略目录中的固定版本：

- `nvdiffrast`: `253ac4fcea7de5f396371124af597e6cc957bfae`
- `nvdiffmodeling`: `9b2ba2eff83c7d90127f78c20773b06ddc3ae1db`

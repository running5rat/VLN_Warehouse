// Headless Recast 1.6.0 build and Detour query, based on Sample_SoloMesh.
#include "Recast.h"
#include "DetourNavMesh.h"
#include "DetourNavMeshBuilder.h"
#include "DetourNavMeshQuery.h"
#include "DetourAlloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
static void world(std::ostream& o, const float* p) { o << '[' << p[0] << ',' << -p[2] << ',' << p[1] << ']'; }
static void cells(const std::string& path, const rcCompactHeightfield& h) {
    std::ofstream out(path); out << "x,y,z,region\n";
    for (int z=0; z<h.height; ++z) for(int x=0; x<h.width; ++x) {
        auto cell=h.cells[x+z*h.width];
        for(unsigned i=cell.index; i<cell.index+cell.count; ++i) if(h.areas[i]!=RC_NULL_AREA)
            out << h.bmin[0]+(x+.5f)*h.cs << ',' << -(h.bmin[2]+(z+.5f)*h.cs) << ','
                << h.bmin[1]+h.spans[i].y*h.ch << ',' << h.spans[i].reg << '\n';
    }
}
int main(int argc, char** argv) {
 try {
    if(argc!=5) throw std::runtime_error("usage: build_navmesh scene.bin output_dir radius queries.txt");
    auto started=std::chrono::steady_clock::now();
    std::string dir=argv[2]; float radius=std::stof(argv[3]);
    std::ifstream in(argv[1],std::ios::binary);
    unsigned counts[2]; float bounds[6];
    require(bool(in.read((char*)counts,8)),"mesh header");
    require(bool(in.read((char*)bounds,24)),"mesh bounds");
    require(counts[0]>0 && counts[1]>0 && counts[0]<50000000 && counts[1]<50000000,"mesh size");
    std::vector<float> verts(3*counts[0]); std::vector<int> tris(3*counts[1]);
    require(bool(in.read((char*)verts.data(),verts.size()*4)),"vertices");
    require(bool(in.read((char*)tris.data(),tris.size()*4)),"triangles");
    for(int i:tris) require(i>=0 && (unsigned)i<counts[0],"triangle index");
    rcConfig cfg{};
    cfg.cs=.1f; cfg.ch=.05f; cfg.walkableSlopeAngle=15;
    cfg.walkableHeight=(int)ceil(1.6f/cfg.ch); cfg.walkableClimb=(int)floor(.10f/cfg.ch);
    cfg.walkableRadius=(int)ceil(radius/cfg.cs); cfg.maxEdgeLen=120;
    cfg.maxSimplificationError=1.0f; cfg.minRegionArea=16; cfg.mergeRegionArea=64;
    cfg.maxVertsPerPoly=6; cfg.detailSampleDist=.6f; cfg.detailSampleMaxError=.05f;
    std::copy(bounds,bounds+3,cfg.bmin); std::copy(bounds+3,bounds+6,cfg.bmax);
    rcCalcGridSize(cfg.bmin,cfg.bmax,cfg.cs,&cfg.width,&cfg.height);
    rcContext ctx;
    auto hf=rcAllocHeightfield(); require(hf,"allocate heightfield");
    require(rcCreateHeightfield(&ctx,*hf,cfg.width,cfg.height,cfg.bmin,cfg.bmax,cfg.cs,cfg.ch),"create heightfield");
    std::vector<unsigned char> areas(counts[1],RC_NULL_AREA);
    rcMarkWalkableTriangles(&ctx,cfg.walkableSlopeAngle,verts.data(),counts[0],tris.data(),counts[1],areas.data());
    // This experiment is warehouse ground-floor navigation, not shelf-top navigation.
    for(unsigned t=0;t<counts[1];++t)
        if(std::max({verts[tris[t*3]*3+1],verts[tris[t*3+1]*3+1],verts[tris[t*3+2]*3+1]})>.15f)
            areas[t]=RC_NULL_AREA;
    require(rcRasterizeTriangles(&ctx,verts.data(),counts[0],tris.data(),areas.data(),counts[1],*hf,cfg.walkableClimb),"rasterize");
    rcFilterLowHangingWalkableObstacles(&ctx,cfg.walkableClimb,*hf);
    rcFilterLedgeSpans(&ctx,cfg.walkableHeight,cfg.walkableClimb,*hf);
    rcFilterWalkableLowHeightSpans(&ctx,cfg.walkableHeight,*hf);
    auto chf=rcAllocCompactHeightfield(); require(chf,"allocate compact field");
    require(rcBuildCompactHeightfield(&ctx,cfg.walkableHeight,cfg.walkableClimb,*hf,*chf),"compact");
    rcFreeHeightField(hf);
    cells(dir+"/walkable-before.csv",*chf);
    require(rcErodeWalkableArea(&ctx,cfg.walkableRadius,*chf),"radius erosion");
    require(rcBuildDistanceField(&ctx,*chf),"distance field");
    require(rcBuildRegions(&ctx,*chf,0,cfg.minRegionArea,cfg.mergeRegionArea),"regions");
    cells(dir+"/walkable-after.csv",*chf);
    auto contours=rcAllocContourSet(); require(contours,"allocate contours");
    require(rcBuildContours(&ctx,*chf,cfg.maxSimplificationError,cfg.maxEdgeLen,*contours),"contours");
    auto mesh=rcAllocPolyMesh(); require(mesh,"allocate polygons");
    require(rcBuildPolyMesh(&ctx,*contours,cfg.maxVertsPerPoly,*mesh),"polygons");
    require(mesh->npolys>0,"empty navmesh");
    auto detail=rcAllocPolyMeshDetail(); require(detail,"allocate detail");
    require(rcBuildPolyMeshDetail(&ctx,*mesh,*chf,cfg.detailSampleDist,cfg.detailSampleMaxError,*detail),"detail mesh");
    for(int i=0;i<mesh->npolys;++i) {mesh->flags[i]=1; mesh->areas[i]=0;}
    dtNavMeshCreateParams params{};
    params.verts=mesh->verts; params.vertCount=mesh->nverts;
    params.polys=mesh->polys; params.polyAreas=mesh->areas; params.polyFlags=mesh->flags;
    params.polyCount=mesh->npolys; params.nvp=mesh->nvp;
    params.detailMeshes=detail->meshes; params.detailVerts=detail->verts; params.detailVertsCount=detail->nverts;
    params.detailTris=detail->tris; params.detailTriCount=detail->ntris;
    params.walkableHeight=1.6f; params.walkableRadius=radius; params.walkableClimb=.1f;
    std::copy(mesh->bmin,mesh->bmin+3,params.bmin); std::copy(mesh->bmax,mesh->bmax+3,params.bmax);
    params.cs=cfg.cs; params.ch=cfg.ch; params.buildBvTree=true;
    unsigned char* data=nullptr; int size=0;
    require(dtCreateNavMeshData(&params,&data,&size),"Detour data");
    {std::ofstream file(dir+"/navmesh.bin",std::ios::binary); file.write((char*)data,size); require(bool(file),"save Detour data");}
    // Queries deliberately use bytes reloaded from disk.
    std::vector<unsigned char> saved(size);
    {std::ifstream file(dir+"/navmesh.bin",std::ios::binary); require(bool(file.read((char*)saved.data(),size)),"reload data");}
    require(std::memcmp(data,saved.data(),size)==0,"serialized bytes differ");
    dtFree(data); data=(unsigned char*)dtAlloc(size,DT_ALLOC_PERM); require(data,"reload allocation");
    std::memcpy(data,saved.data(),size);
    auto nav=dtAllocNavMesh(); require(nav,"allocate Detour navmesh");
    require(dtStatusSucceed(nav->init(data,size,DT_TILE_FREE_DATA)),"reload Detour navmesh");
    auto query=dtAllocNavMeshQuery(); require(query,"allocate query");
    require(dtStatusSucceed(query->init(nav,16384)),"query init");
    std::ofstream json(dir+"/navmesh.json"),obj(dir+"/navmesh.obj");
    json << "{\"format\":\"recast-polygon-mesh-v1\",\"coordinate_frame\":\"Isaac world Z-up metres\",\"vertices\":[";
    for(int i=0;i<mesh->nverts;++i) {
        float v[3]={mesh->bmin[0]+mesh->verts[i*3]*cfg.cs,mesh->bmin[1]+mesh->verts[i*3+1]*cfg.ch,mesh->bmin[2]+mesh->verts[i*3+2]*cfg.cs};
        if(i) json<<','; world(json,v); obj<<"v "<<v[0]<<' '<<-v[2]<<' '<<v[1]<<'\n';
    }
    json << "],\"polygons\":[";
    for(int i=0;i<mesh->npolys;++i) {
        if(i)json<<','; json<<"{\"id\":"<<i<<",\"region\":"<<mesh->regs[i]<<",\"vertices\":["; obj<<"f";
        auto poly=&mesh->polys[i*2*mesh->nvp]; int n=0;
        for(;n<mesh->nvp && poly[n]!=RC_MESH_NULL_IDX;++n){if(n)json<<',';json<<poly[n];obj<<' '<<poly[n]+1;}
        obj<<'\n'; json<<"],\"neighbors\":[";
        for(int j=0;j<n;++j){if(j)json<<','; auto v=poly[mesh->nvp+j];json<<(v==RC_MESH_NULL_IDX?-1:(int)v);}
        json<<"]}";
    }
    json<<"]}\n";
    dtQueryFilter filter; filter.setIncludeFlags(1);
    std::ifstream requests(argv[4]); require(bool(requests),"queries file");
    std::ofstream results(dir+"/queries.json"); results<<"["; std::string name; int qi=0, complete=0;
    float a[3],b[3],aw[3],bw[3];
    while(requests>>name>>aw[0]>>aw[1]>>aw[2]>>bw[0]>>bw[1]>>bw[2]) {
        a[0]=aw[0];a[1]=aw[2];a[2]=-aw[1];b[0]=bw[0];b[1]=bw[2];b[2]=-bw[1];
        float ext[3]={.5f,1.f,.5f},na[3]{},nb[3]{};dtPolyRef ar=0,br=0;
        auto sa=query->findNearestPoly(a,ext,&filter,&ar,na);auto sb=query->findNearestPoly(b,ext,&filter,&br,nb);
        dtPolyRef corridor[2048];int npath=0;float straight[2048*3];int nstraight=0;
        dtStatus status=DT_FAILURE,st=DT_FAILURE;
        if(dtStatusSucceed(sa)&&dtStatusSucceed(sb)&&ar&&br) {
            status=query->findPath(ar,br,na,nb,&filter,corridor,&npath,2048);
            if(npath)st=query->findStraightPath(na,nb,corridor,npath,straight,nullptr,nullptr,&nstraight,2048);
        }
        bool ok=dtStatusSucceed(status)&&dtStatusSucceed(st)&&npath>0&&corridor[npath-1]==br&&!dtStatusDetail(status,DT_PARTIAL_RESULT)&&!dtStatusDetail(status,DT_BUFFER_TOO_SMALL)&&!dtStatusDetail(st,DT_BUFFER_TOO_SMALL);
        if(ok)++complete;
        if(qi++)results<<',';results<<"{\"name\":\""<<name<<"\",\"complete\":"<<(ok?"true":"false")<<",\"status\":"<<status<<",\"start_ref\":"<<ar<<",\"end_ref\":"<<br<<",\"requested_start\":";world(results,a);
        results<<",\"requested_end\":";world(results,b);results<<",\"snapped_start\":";world(results,na);results<<",\"snapped_end\":";world(results,nb);
        results<<",\"corridor_polygon_ids\":[";
        for(int j=0;j<npath;++j){const dtMeshTile*tile;const dtPoly*poly;nav->getTileAndPolyByRef(corridor[j],&tile,&poly);if(j)results<<',';results<<(poly-tile->polys);}
        results<<"],\"straight_path\":[";for(int j=0;j<nstraight;++j){if(j)results<<',';world(results,&straight[j*3]);}results<<"]}";
    }
    results<<"]\n";
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ofstream report(dir+"/build-report.json");
    report<<"{\"recast_version\":\"1.6.0\",\"radius_m\":"<<radius<<",\"effective_erosion_radius_m\":"<<cfg.walkableRadius*cfg.cs<<",\"height_m\":1.6,\"max_climb_m\":0.1,\"max_slope_deg\":15,\"cell_size_m\":0.1,\"cell_height_m\":0.05,\"max_walkable_world_z_m\":0.15,\"min_region_cells\":16,\"merge_region_cells\":64,\"max_simplification_error_voxels\":1.0,\"max_vertices_per_polygon\":6,\"grid_width\":"<<cfg.width<<",\"grid_height\":"<<cfg.height<<",\"input_triangles\":"<<counts[1]<<",\"navmesh_vertices\":"<<mesh->nverts<<",\"navmesh_polygons\":"<<mesh->npolys<<",\"detail_triangles\":"<<detail->ntris<<",\"queries\":"<<qi<<",\"complete_queries\":"<<complete<<",\"disk_reload_passed\":true,\"elapsed_s\":"<<elapsed<<"}\n";
    std::cout<<"polygons="<<mesh->npolys<<" queries="<<complete<<'/'<<qi<<" seconds="<<elapsed<<'\n';
    dtFreeNavMeshQuery(query);dtFreeNavMesh(nav);rcFreePolyMeshDetail(detail);rcFreePolyMesh(mesh);rcFreeContourSet(contours);rcFreeCompactHeightfield(chf);
    return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
